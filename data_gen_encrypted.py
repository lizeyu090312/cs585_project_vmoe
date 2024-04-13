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
    

def get_dataset(test_or_train, aggregate_batch_size, maj_perc,
                class_idx_dir="/home/zl310/cs585_project/vmoe/chosen_class_idx_encrypted", 
                save_dir="/home/zl310/cs585_project/vmoe/ImageNetData_64classes_enc", count=True):
    n_classes = 64  # only generate the data for 64 classes, n_classes can be sampled easily
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
            batch_size=2048,#dataset_config_test.batch_size,
            # batch_size shouldn't matter here, since we're simply collecting the 64 classes for more efficient future processing
            process=dataset_config_test.process, 
        )
    elif test_or_train == "train":
        dataset_config_train = model_config.dataset.train
        dataset = input_pipeline.get_dataset(
            variant='train',
            name=dataset_config_train.name, 
            split=dataset_config_train.split, 
            batch_size=2048,#dataset_config_train.batch_size, 
            process=dataset_config_train.process
        )
    
    maj_class_f = open(os.path.join(class_idx_dir, f"maj_class.txt"), 'r')
    lines = maj_class_f.readlines()
    maj_class = int(lines[0])
    num_samples_of_maj_class = int(lines[1])
    maj_class_f.close()

    count_maj_class = 0
    class_set.remove(maj_class)
    class_list = list(class_set)
    
    total_maj_so_far = 0
    max_possible_maj = int(num_samples_of_maj_class - (num_samples_of_maj_class % (aggregate_batch_size * maj_perc)))
    print(max_possible_maj)
    aggr_obj = AggregateTensor(aggregate_batch_size, class_list, maj_class, maj_perc=maj_perc)

    ncorrect, ntotal, i = 0, 0, 0
    for batch in dataset:
        mask = batch['__valid__']
        orig_img = batch['image']
        if jnp.sum(mask) != orig_img.shape[0]:
            continue
        orig_true_lbl = np.array(jnp.argmax(batch['labels'], axis=1))
        img, true_lbl, temp_aggr_obj = aggr_obj.update(orig_img, orig_true_lbl)

        if count:
            count_maj_class += np.sum(true_lbl == maj_class)
            print(count_maj_class)
        elif img is not None:
            maj_now = np.sum(true_lbl == maj_class)
            print(f"number of classes from maj class {maj_now}")
            total_maj_so_far += maj_now
            if total_maj_so_far >= max_possible_maj:
                print(f"reached max_possible_maj, max_possible_maj = {max_possible_maj}, total_maj_so_far = {total_maj_so_far}")
                break
            logits, _, indices_distr = model.apply({'params': checkpoint}, img)
            log_p = jax.nn.log_softmax(logits)
            preds = jnp.argmax(log_p, axis=1)
            aggr_obj = temp_aggr_obj
            ncorrect += jnp.sum(preds == true_lbl)
            ntotal += len(true_lbl)
            print(f'Test accuracy, iteration {i}: {ncorrect / ntotal * 100:.2f}%')
            i += 1
    if count == True:
        with open(os.path.join(class_idx_dir, f"maj_class.txt"), 'a') as f_ptr:
            f_ptr.write(f"{count_maj_class}\n")
    return

get_dataset(test_or_train="train", aggregate_batch_size=32, count=False, maj_perc=0.25)