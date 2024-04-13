"""
Generate data for unencrypted channel. 
The output should be stored in ./expert_load_train_ImageNetData_enc or 
./expert_load_test_ImageNetData_enc. There are only two files:
x, y (x has shape (num_batches, 8)), and y (of shape
(num_batches,)). x stores the number of tiles each expert receives
in batch b (e.g. x[0, :] stores # tiles per expert in batch 0). 
y stores the majority class within each batch. 
"""
from tqdm import tqdm
import os
import logging
import argparse
import collections


class MyCounter(collections.Counter):
    def __str__(self):
        return ",".join('{}:{}'.format(int(k), int(v)) for k, v in self.items())

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from vmoe.nn import models
from vmoe.data import input_pipeline
from vmoe.checkpoints import partitioned
from vmoe.configs.vmoe_paper.vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012 import get_config, IMAGE_SIZE, BATCH_SIZE
import jax.numpy as jnp
import jax

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


class AggregateTensor:
    def __init__(self, agg_bs: int, which_classes: list[int], maj_class: int, maj_perc: float):
        self.aggregate_batch_size = agg_bs
        self.maj_class = maj_class
        self.maj_perc = maj_perc
        self.which_classes = which_classes
        assert maj_class not in which_classes
        assert agg_bs % 2 == 0
        self.maj_sz = int(maj_perc * agg_bs)
        self.maj_img = -1 * np.ones((self.maj_sz, IMAGE_SIZE, IMAGE_SIZE, 3))
        self.other_sz = int(agg_bs - int(maj_perc * agg_bs))
        self.other_img = -1 * np.ones((self.other_sz, IMAGE_SIZE, IMAGE_SIZE, 3))
        self.maj_num_so_far = 0
        self.other_num_so_far = 0

        self.each_class_count = {c: 0 for c in which_classes}  
        # records num so far see for each class in which_classes (not including maj_class)
        self.max_other_freq = self.maj_sz - 1
        self.maj_label = -1 * np.ones(self.maj_sz)
        self.other_label = -1 * np.ones(self.other_sz)
        return
    
    def update(self, new_img, new_label):
        mask_maj = new_label == self.maj_class
        mask_other = np.isin(np.array(new_label), self.which_classes)
        img_maj, lbl_maj = new_img[mask_maj], new_label[mask_maj]
        img_other, lbl_other = new_img[mask_other], new_label[mask_other]
        lbl_to_idx = dict()  # records locations for finding images of cls = x, i do not want to use defaultdict
        for idx, lbl_o in enumerate(lbl_other):
            if lbl_o not in lbl_to_idx.keys():
                lbl_to_idx[lbl_o] = []
            lbl_to_idx[lbl_o].append(idx)

        how_many_more_maj = min(int(self.maj_sz - self.maj_num_so_far), img_maj.shape[0])
        if how_many_more_maj > 0:
            self.maj_img[self.maj_num_so_far:self.maj_num_so_far+how_many_more_maj] = img_maj[0:how_many_more_maj]
            self.maj_label[self.maj_num_so_far:self.maj_num_so_far+how_many_more_maj] = lbl_maj[0:how_many_more_maj]
            self.maj_num_so_far += how_many_more_maj

        how_many_more_other = min(int(self.other_sz - self.other_num_so_far), img_other.shape[0])
        max_class = None
        while how_many_more_other > 0:
            for cls in self.which_classes:
                if cls in lbl_to_idx.keys() and len(lbl_to_idx[cls]) > 0:  # if this class exists inside of this label iteration
                    for iidx in lbl_to_idx[cls]:
                        if self.each_class_count[cls] < self.max_other_freq and how_many_more_other > 0:
                            self.other_img[self.other_num_so_far] = img_other[iidx]
                            self.other_label[self.other_num_so_far] = lbl_other[iidx]
                            self.other_num_so_far += 1
                            self.each_class_count[cls] += 1
                            how_many_more_other -= 1
                        if self.each_class_count[cls] == self.max_other_freq:
                            max_class = cls
            break  # cycled through all the classes, but did not meet quota due to one (or more) other classes filling up, 
            # which means cannot add more classes -> break
        if how_many_more_other > 0:
            logging.info(f"Did not use up all other data, how_many_more_other={how_many_more_other}, filled {max_class} (other cls")
        if self.maj_num_so_far == self.maj_sz and self.other_num_so_far == self.other_sz:
            shuffle_idx = np.random.permutation(range(self.aggregate_batch_size))
            img_out = np.concatenate((self.maj_img, self.other_img), axis=0)[shuffle_idx]
            label_out = np.concatenate((self.maj_label, self.other_label))[shuffle_idx]
            new_agg_obj = AggregateTensor(self.aggregate_batch_size, self.which_classes, self.maj_class, self.maj_perc)
            return img_out, label_out, new_agg_obj
        return None, None, None
    
    def update_force(self):
        """
        returns img and label with correct proportions, does not guarantee that ret.shape[0] >= 32
        """
        other_num_using_maj_num = int(self.maj_num_so_far * (1-self.maj_perc) / self.maj_perc)
        maj_num_using_other_num = int(self.other_num_so_far * self.maj_perc / (1-self.maj_perc))
        if self.maj_num_so_far > maj_num_using_other_num:
            if self.other_num_so_far + maj_num_using_other_num > 0:
                shuffle_idx = np.random.permutation(range(self.other_num_so_far + maj_num_using_other_num))
                logging.info(f"update_force forced out {len(shuffle_idx)} images")
                other_img = self.other_img[0:self.other_num_so_far]
                other_label = self.other_label[0:self.other_num_so_far]

                maj_img = self.maj_img[0:maj_num_using_other_num]
                maj_label = self.maj_label[0:maj_num_using_other_num]
                return np.concatenate((maj_img, other_img), axis=0)[shuffle_idx], \
                    np.concatenate((maj_label, other_label), axis=0)[shuffle_idx], None
            else: 
                return None, None, None
        else:
            if other_num_using_maj_num + self.maj_num_so_far > 0:
                shuffle_idx = np.random.permutation(range(other_num_using_maj_num + self.maj_num_so_far))
                # print(len(shuffle_idx))
                other_img = self.other_img[0:other_num_using_maj_num]
                other_label = self.other_label[0:other_num_using_maj_num]
                maj_img = self.maj_img[0:self.maj_num_so_far]
                maj_label = self.maj_label[0:self.maj_num_so_far]
                # print(np.concatenate((maj_img, other_img), axis=0).shape)
                return np.concatenate((maj_img, other_img), axis=0)[shuffle_idx], \
                    np.concatenate((maj_label, other_label), axis=0)[shuffle_idx], None
            else:
                return None, None, None


def highest_power_of_2(n):
    res = 0
    for i in range(n, 0, -1):
        # if i is a power of 2
        if (i & (i - 1)) == 0:
            res = i
            break
    return res


def process_indices(indices_distr, maj_class, save_name):
    # indices_distr has shape (8, 55808, 512) for batch_size = 1024
    expert_load_list = [[], []]  # first list for layer5, second list for layer7
    num_experts = 8
    for idx, layer in enumerate([indices_distr['idx_5'], indices_distr['idx_7']]):
        for exp in range(num_experts):
            expert_load_list[idx].append(np.sum(np.array(layer[exp, :, 0]).flatten() != 0))
    with open(os.path.join(save_name), 'a') as f_ptr:
        for i in range(len(expert_load_list)):
            f_ptr.write(",".join(str(n) for n in expert_load_list[i]))
            f_ptr.write(",")
        f_ptr.write(f"{maj_class}\n")
    return


def poll_aggr_obj(aggr_objs: dict[int, AggregateTensor], 
                  class_to_max_possible: dict[int, int], 
                  class_to_so_far: dict[int, int], new_img, new_label, 
                  force=False):
    # class_to_so_far is the number of inputs that have been processed for this majority
    # class so far (each class can be a majority class)
    # returns True when needs to break
    img_dict, label_dict = dict(), dict()
    if force == True:
        for maj_cls, aggr_obj in aggr_objs.items():
            img_out, label_out, _ = aggr_obj.update_force()
            if label_out is not None and len(label_out) >= 32:
                truncated_len = highest_power_of_2(len(label_out))
                img_dict[maj_cls] = img_out[0:truncated_len]
                label_dict[maj_cls] = label_out[0:truncated_len]
        return img_dict, label_dict, None
    done = 0
    for maj_cls, aggr_obj in aggr_objs.items():
        img_out, label_out, new_agg_obj = aggr_obj.update(new_img, new_label)
        if class_to_so_far[maj_cls] >= class_to_max_possible[maj_cls]:
            done += 1
        if img_out is not None:
            img_dict[maj_cls] = img_out
            label_dict[maj_cls] = label_out
            aggr_objs[maj_cls] = new_agg_obj
            class_to_so_far[maj_cls] += np.sum(label_out == maj_cls)
    if done == len(class_to_max_possible.keys()):
        return None, None, True
    return img_dict, label_dict, False


def eval_and_log(model, checkpoint, img_d, lbl_d, ncorrect, ntotal, maj_perc,
                 save_dir, num_classes):
    logged = False
    for maj_cls in img_d.keys():
        img, lbl = img_d[maj_cls], lbl_d[maj_cls]
        if logged == False:
            c = MyCounter(np.array(lbl, dtype=np.int32))
            logging.info(f"{c.most_common()}")
            logged = True
        logits, _, indices_distr = model.apply({'params': checkpoint}, img)
        preds = jnp.argmax(jax.nn.log_softmax(logits), axis=1)
        ncorrect += jnp.sum(preds == lbl)
        ntotal += len(lbl)
        assert int(maj_perc * len(lbl)) == np.sum(maj_cls == lbl)
        perc_str_formatted = "_".join(str(maj_perc).split("."))
        save_name = os.path.join(f"{save_dir}", f"n_class_{num_classes}_maj_perc_{perc_str_formatted}.txt")
        if os.path.isfile(save_name) == False:
            with open(save_name, 'w') as f_ptr:
                f_ptr.write(",".join(f"l5e{i+1}" for i in range(8)))
                f_ptr.write(",")
                f_ptr.write(",".join(f"l7e{i+1}" for i in range(8)))
                f_ptr.write(",maj_class\n")
        process_indices(indices_distr=indices_distr, maj_class=maj_cls, save_name=save_name)
    return ncorrect, ntotal


def get_dataset(model, checkpoint, model_config, test_or_train, aggregate_batch_size, maj_perc, n_classes,
                class_idx_dir="/home/zl310/cs585_project/vmoe/chosen_class_idx_encrypted", count=False):
    assert float(maj_perc) > float(1 / n_classes), f"chekc maj_perc, maj_perc={maj_perc}, (1 / n_classes)={1 / n_classes}"
    class_set = set(np.load(os.path.join(class_idx_dir, f"n_{n_classes}.npy")))
    save_dir = f"/home/zl310/cs585_project/vmoe/expert_load_{test_or_train}_ImageNetData_enc"

    if test_or_train == "test":
        dataset_config_test = model_config.dataset.test
        dataset = input_pipeline.get_dataset(variant='test', name=dataset_config_test.name, split=dataset_config_test.split, 
            batch_size=128, process=dataset_config_test.process)
        # batch_size shouldn't matter too much here, since we're simply collecting 
        # the 64 classes for more efficient future processing. small batch size here
        # means less overflow for each aggregate object
    elif test_or_train == "train":
        dataset_config_train = model_config.dataset.train
        dataset = input_pipeline.get_dataset(variant='train', name=dataset_config_train.name, split=dataset_config_train.split, 
            batch_size=512, process=dataset_config_train.process)  #batch_size=dataset_config_train.batch_size
    else:
        raise Exception(f"check test_or_train, test_or_train = {test_or_train}")
    
    class_to_max_possible, class_to_freq, maj_class_to_aggr_obj, class_to_so_far = dict(), dict(), dict(), dict()
    with open(os.path.join(class_idx_dir, f"class_count.txt"), 'r') as f_ptr:
        lines = f_ptr.readlines()
        for line in lines:  
            # creates class to max possible # dict (ensures every batch has aggregate_batch_size*maj_perc 
            # samples of majority class), freq dict of each class, and aggregate object where each class is
            # majority class, and number of inputs processed for this class so far in class_to_so_far
            parts = line.strip().split(':')
            cls, freq = int(parts[0]), int(parts[1])
            if cls in class_set:  # only creating the dicts for the classes considered (based on n_classes)
                class_list = list(class_set)
                class_list.remove(cls)
                if test_or_train == "train":
                    class_to_freq[cls] = freq
                    class_to_max_possible[cls] = int(freq - (freq % (aggregate_batch_size * maj_perc)))
                else:
                    freq = 50
                    class_to_freq[cls] = freq
                    class_to_max_possible[cls] = int(freq - (freq % (aggregate_batch_size * maj_perc)))
                class_to_so_far[cls] = 0
                maj_class_to_aggr_obj[cls] = AggregateTensor(aggregate_batch_size, class_list, maj_class=cls, maj_perc=maj_perc)
    assert len(class_to_max_possible.keys()) == n_classes
    assert len(class_to_max_possible.keys()) == len(class_to_freq.keys()) == \
            len(maj_class_to_aggr_obj.keys()) == len(class_to_so_far.keys())
    which_classes64 = np.load(os.path.join(class_idx_dir, "n_64.npy"))
    count_maj_class = {c: 0 for c in which_classes64}
    
    # max_possible_maj = int(num_samples_of_maj_class - (num_samples_of_maj_class % (aggregate_batch_size * maj_perc)))
    # aggr_obj = AggregateTensor(aggregate_batch_size, class_list, maj_class, maj_perc=maj_perc)

    ncorrect, ntotal, i = 0, 0, 0
    for batch in tqdm(dataset):
        mask = batch['__valid__']
        orig_img = batch['image']
        if jnp.sum(mask) != orig_img.shape[0]:
            continue
        orig_true_lbl = np.array(jnp.argmax(batch['labels'], axis=1))
        img_d, lbl_d, break_cond = poll_aggr_obj(maj_class_to_aggr_obj, class_to_max_possible, 
                                                 class_to_so_far, orig_img, orig_true_lbl)
        if count:
            for c in which_classes64:
                count_maj_class[c] += np.sum(orig_true_lbl == c)
        elif break_cond == True:
            break
        elif len(img_d.keys()) > 0:
            ncorrect, ntotal = eval_and_log(model=model, checkpoint=checkpoint, img_d=img_d, 
                                            lbl_d=lbl_d, ncorrect=ncorrect, ntotal=ntotal, 
                                            maj_perc=maj_perc, save_dir=save_dir, num_classes=n_classes)
            
            logging.info(f'Test accuracy, iteration {i}: {ncorrect / ntotal * 100:.2f}%')
            i += 1
    # after main for loop, checks for left-over data
    if count == False:
        # poll_aggr_obj ensures the data batch size is at least 32
        img_d, lbl_d, break_cond = poll_aggr_obj(maj_class_to_aggr_obj, class_to_max_possible, 
                                                 class_to_so_far, orig_img, orig_true_lbl, force=True)
        if len(img_d.keys()) > 0:
            ncorrect, ntotal = eval_and_log(model=model, checkpoint=checkpoint, img_d=img_d, 
                                            lbl_d=lbl_d, i=i, ncorrect=ncorrect, ntotal=ntotal, 
                                            maj_perc=maj_perc, save_dir=save_dir, num_classes=n_classes)
    elif count == True:
        with open(os.path.join(class_idx_dir, f"class_count.txt"), 'a') as f_ptr:
            for c, freq in count_maj_class.items():
                f_ptr.write(f"{c}:{freq}\n")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_or_test', type=str, required=True)
    parser.add_argument('-p', '--maj_perc', type=float, required=False)
    parser.add_argument('-n', '--n_classes', type=int, required=False)
    args = parser.parse_args()
    if args.train_or_test == "test":
        logging.basicConfig(filename=f'vmoe_datagen_{args.train_or_test}_enc.log', 
                            level=logging.INFO, 
                            format='%(levelname)s: %(asctime)s %(message)s')
    elif args.train_or_test == "train":
        logging.basicConfig(filename=f'/home/zl310/cs585_project/vmoe/vmoe_datagen_train_enc_logs/vmoe_datagen_{args.train_or_test}_{args.maj_perc}_{args.n_classes}_enc.log', 
                            level=logging.INFO, 
                            format='%(levelname)s: %(asctime)s %(message)s')
    model_config = get_config()
    model_cls = getattr(models, model_config.model.name)
    model = model_cls(deterministic=True, **model_config.model)

    # using this model: 'gs://vmoe_checkpoints/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012'
    checkpoint_prefix = 'vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012'
    checkpoint = partitioned.restore_checkpoint(prefix=checkpoint_prefix, tree=None)
    maj_perc_list = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    n_class_list = [2, 4, 8, 16, 32, 64]
    agg_bs = 32
    if args.train_or_test == "test":
        for n_classes in n_class_list:
            for maj_perc in maj_perc_list:
                if maj_perc > 1 / n_classes:
                    logging.info("-"*10)
                    logging.info("-"*10)
                    logging.info(f"starting n_classes = {int(n_classes)}, maj_perc = {maj_perc}")
                    get_dataset(model=model, checkpoint=checkpoint, model_config=model_config, 
                                test_or_train=args.train_or_test, aggregate_batch_size=agg_bs, count=False, 
                                maj_perc=maj_perc, n_classes=n_classes)
    elif args.train_or_test == "train":
        get_dataset(model=model, checkpoint=checkpoint, model_config=model_config, 
                    test_or_train=args.train_or_test, aggregate_batch_size=agg_bs, count=False, 
                    maj_perc=args.maj_perc, n_classes=args.n_classes)
    else:
        raise Exception(f"check args.train_or_test, {args.train_or_test}")
        