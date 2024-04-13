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

        self.maj_label = -1 * np.ones(self.maj_sz)
        self.other_label = -1 * np.ones(self.other_sz)
        return
    
    def update(self, new_img, new_label):
        mask_maj = new_label == self.maj_class
        mask_other = np.isin(np.array(new_label), self.which_classes)
        img_maj, lbl_maj = new_img[mask_maj], new_label[mask_maj]
        img_other, lbl_other = new_img[mask_other], new_label[mask_other]
        how_many_more_maj = min(int(self.maj_sz - self.maj_num_so_far), img_maj.shape[0])
        if how_many_more_maj > 0:
            self.maj_img[self.maj_num_so_far:self.maj_num_so_far+how_many_more_maj] = img_maj[0:how_many_more_maj]
            self.maj_label[self.maj_num_so_far:self.maj_num_so_far+how_many_more_maj] = lbl_maj[0:how_many_more_maj]
            self.maj_num_so_far += how_many_more_maj

        how_many_more_other = min(int(self.other_sz - self.other_num_so_far), img_other.shape[0])
        if how_many_more_other > 0:
            self.other_img[self.other_num_so_far:self.other_num_so_far+how_many_more_other] = img_other[0:how_many_more_other]
            self.other_label[self.other_num_so_far:self.other_num_so_far+how_many_more_other] = lbl_other[0:how_many_more_other]
            self.other_num_so_far += how_many_more_other
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
            shuffle_idx = np.random.permutation(range(self.other_num_so_far + maj_num_using_other_num))
            other_img = self.other_img[0:self.other_num_so_far]
            other_label = self.other_label[0:self.other_num_so_far]

            maj_img = self.maj_img[0:maj_num_using_other_num]
            maj_label = self.maj_label[0:maj_num_using_other_num]
            print(len(other_label))
            print(len(maj_label))
            return np.concatenate((maj_img, other_img), axis=0)[shuffle_idx], \
                np.concatenate((maj_label, other_label), axis=0)[shuffle_idx], None
        else:
            shuffle_idx = np.random.permutation(range(other_num_using_maj_num + self.maj_num_so_far))
            other_img = self.other_img[0:other_num_using_maj_num]
            other_label = self.other_label[0:other_num_using_maj_num]
            print(len(other_label))
            maj_img = self.maj_img[0:self.maj_num_so_far]
            maj_label = self.maj_label[0:self.maj_num_so_far]
            print(len(maj_label))
            return np.concatenate((maj_img, other_img), axis=0)[shuffle_idx], \
                np.concatenate((maj_label, other_label), axis=0)[shuffle_idx], None


def highest_power_of_2(n):
    res = 0
    for i in range(n, 0, -1):
        # if i is a power of 2
        if (i & (i - 1)) == 0:
            res = i
            break
    return res


def process_indices(indices_distr, save_dir, num_classes, maj_class):
    # indices_distr has shape (8, 55808, 512) for batch_size = 1024
    expert_load_list = [[], []]  # first list for layer5, second list for layer7
    num_experts = 8
    for idx, layer in enumerate([indices_distr['idx_5'], indices_distr['idx_7']]):
        for exp in range(num_experts):
            expert_load_list[idx].append(np.sum(np.array(layer[exp]).flatten() != 0))


    return


def poll_aggr_obj(aggr_objs: dict[int, AggregateTensor], 
                  class_to_max_possible: dict[int, int], 
                  class_to_so_far: dict[int, int], new_img, new_label):
    # class_to_so_far is the number of inputs that have been processed for this majority
    # class so far (each class can be a majority class)
    # returns True when needs to break
    img_dict, label_dict = dict(), dict()
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


def eval_and_log(model, checkpoint, img_d, lbl_d, i, ncorrect, ntotal, maj_perc,
                 save_dir, num_classes):
    for maj_cls in img_d.keys():
        img, lbl = img_d[maj_cls], lbl_d[maj_cls]
        logits, _, indices_distr = model.apply({'params': checkpoint}, img)
        preds = jnp.argmax(jax.nn.log_softmax(logits), axis=1)
        ncorrect += jnp.sum(preds == lbl)
        ntotal += len(lbl)
        assert int(maj_perc * len(lbl)) == np.sum(maj_cls == lbl)
        # logging.info(f'Test accuracy, iteration {i}: {ncorrect / ntotal * 100:.2f}%')
        print(f'Test accuracy, iteration {i}: {ncorrect / ntotal * 100:.2f}%')
        process_indices(indices_distr, save_dir, num_classes, maj_cls)
    return ncorrect, ntotal


def get_dataset(test_or_train, aggregate_batch_size, maj_perc, n_classes,
                class_idx_dir="/home/zl310/cs585_project/vmoe/chosen_class_idx_encrypted", 
                save_dir="/home/zl310/cs585_project/vmoe/ImageNetData_64classes_enc", count=False):
    class_set = set(np.load(os.path.join(class_idx_dir, f"n_{n_classes}.npy")))

    model_config = get_config()
    model_cls = getattr(models, model_config.model.name)
    model = model_cls(deterministic=True, **model_config.model)

    # using this model: 'gs://vmoe_checkpoints/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012'
    checkpoint_prefix = 'vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012'
    checkpoint = partitioned.restore_checkpoint(prefix=checkpoint_prefix, tree=None)

    dataset_config_test = model_config.dataset.test

    if test_or_train == "test":
        dataset = input_pipeline.get_dataset(
            variant='test',
            name=dataset_config_test.name, 
            split=dataset_config_test.split, 
            batch_size=128,#dataset_config_test.batch_size,
            # batch_size shouldn't matter too much here, since we're simply collecting 
            # the 64 classes for more efficient future processing
            process=dataset_config_test.process, 
        )
    elif test_or_train == "train":
        dataset_config_train = model_config.dataset.train
        dataset = input_pipeline.get_dataset(
            variant='train',
            name=dataset_config_train.name, 
            split=dataset_config_train.split, 
            batch_size=256,#dataset_config_train.batch_size, 
            process=dataset_config_train.process
        )
    else:
        raise Exception(f"check test_or_train, test_or_train = {test_or_train}")
    
    class_to_max_possible, class_to_freq, maj_class_to_aggr_obj, class_to_so_far = dict(), dict(), dict(), dict()
    with open(os.path.join(class_idx_dir, f"class_count.txt"), 'r') as f_ptr:
        lines = f_ptr.readlines()
        for line in lines:  
            # creates class to max possible # dict (ensures every batch has aggregate_batch_size*maj_perc 
            # samples of majority class), freq dict of each class, and aggregate object where each class is
            # majority class, and number of inputs processed for this class so far class_to_so_far
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
            ncorrect, ntotal = eval_and_log(model, checkpoint, img_d, lbl_d, i, ncorrect, ntotal, maj_perc, 
                                            save_dir, n_classes)
            
            print(f'Test accuracy, iteration {i}: {ncorrect / ntotal * 100:.2f}%')
            i += 1
    img, true_lbl, _ = aggr_obj.update_force()
    if len(true_lbl) >= 32 and count == False:
        truncated_len = highest_power_of_2(len(true_lbl))
        logits, _, indices_distr = model.apply({'params': checkpoint}, img[0:truncated_len])
        # process_indices()
        preds = jnp.argmax(jax.nn.log_softmax(logits), axis=1)
        ncorrect += jnp.sum(preds == true_lbl[0:truncated_len])
        ntotal += len(true_lbl)
    elif count == True:
        with open(os.path.join(class_idx_dir, f"class_count.txt"), 'a') as f_ptr:
            for c, freq in count_maj_class.items():
                f_ptr.write(f"{c}:{freq}\n")
    return

get_dataset(test_or_train="train", aggregate_batch_size=32, count=True, maj_perc=0.25, n_classes=4)