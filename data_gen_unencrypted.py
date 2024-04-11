"""
Generate data for unencrypted channel. 
The output should be stored in ./expert_assign_train_ImageNetData or 
./expert_assign_test_ImageNetData. Each ImageNet batch creates three files:
x_layer5, x_layer7 (both of shape (batch_sz, 12, 12, 2)), and y (of shape
batch_sz). x_layer{i} stores the allocation of each patch to each expert
y[b] contains the class label of each x_layer5[b, :, :, :] in the batch. 
"""
import os
import re
import uuid 
import tqdm
import logging
import argparse
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
from jax import numpy as jnp

import tensorflow as tf

from vmoe.nn import models
from vmoe.data import input_pipeline
from vmoe.checkpoints import partitioned

from vmoe.configs.vmoe_paper.vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012 import get_config, IMAGE_SIZE, BATCH_SIZE

import numpy as np

"""
Adapted from vmoe/notebooks/demo_eee_CIFAR100.ipynb by Michael Li
Structure:
vmoe
    vmoe/
    this notebook
    vit_jax/ (from vision_transformer)
    vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012.data-00000-of-00001
    vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012.index
"""


def process_indices(indices_distr: dict[str: jnp.ndarray], batch_idx: int, mask,
                    class_lbl, fdir, batch_sz=BATCH_SIZE, img_sz: int = int(np.sqrt(144))):
    # img_sz * img_sz == number of tiles/patches in each 384*384 image. There are 144 tiles per image.
    class_lbl = np.array(class_lbl, dtype=np.uint16) * mask  # 2^16 = 65,536 > 1000
    topk = 2
    img5 = np.zeros(shape=(batch_sz, int(img_sz * img_sz), topk), dtype=np.uint8)
    img7 = np.zeros(shape=(batch_sz, int(img_sz * img_sz), topk), dtype=np.uint8)

    exp, n, _ = indices_distr['idx_5'].shape
    for img, layer in zip([img5, img7], [indices_distr['idx_5'], indices_distr['idx_7']]):
        for expert in range(exp):
            for buf in range(n):
                if layer[expert, buf, 0] != 0 and layer[expert, buf, 1] != 0:
                    b_idx, p_idx = int(layer[expert, buf, 0]), int(layer[expert, buf, 1])
                    if img[b_idx-1, p_idx-1, 0] == 0:
                        img[b_idx-1, p_idx-1, 0] = expert + 1  # img[batch_idx, patch_idx] = expert_id, 1-indexing
                    else:
                        img[b_idx-1, p_idx-1, 1] = expert + 1  # those that aren't assigned experts are left with expertID=0
    img5_reshaped = img5.reshape(batch_sz, img_sz, img_sz, topk)
    img7_reshaped = img7.reshape(batch_sz, img_sz, img_sz, topk)
    if jnp.sum(mask) != batch_sz:
        img5_reshaped = img5_reshaped[mask]
        img7_reshaped = img7_reshaped[mask]
        logging.info(f"jnp.sum(mask) != batch_sz for batch_idx = {batch_idx}, jnp.sum(mask) = {jnp.sum(mask)}, batch_sz = {batch_sz}")
    ID = uuid.uuid4()
    np.save(os.path.join(fdir, f"x_{ID}_batch_{batch_idx}_layer_5.npy"), img5_reshaped)
    np.save(os.path.join(fdir, f"x_{ID}_batch_{batch_idx}_layer_7.npy"), img7_reshaped)

    np.save(os.path.join(fdir, f"y_{ID}_batch_{batch_idx}_layer_both.npy"), class_lbl)
    logging.info(f"finished for batch_idx = {batch_idx}")
    # indices_distr has shape (8, 55808, 512) for batch_size = 1024
    return


def which_batches_are_done(dir_abs_path) -> set:
    def extract_last_integer(filename):
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
        else:
            return None
    pattern = r'_batch_(\d+)_layer_both\.npy$'
    files = os.listdir(dir_abs_path)
    ret = set()
    for file in files:
        ret.add(extract_last_integer(file))
    ret.add(None)
    ret.remove(None)
    return ret


def gen_data(model, dataset, checkpoint, save_dir, start_b, end_b):
    ncorrect = 0
    ntotal = 0
    i = 0
    logging.info('-' * 10)
    logging.info('-' * 10)
    if end_b == -1:
        end_b = 10000000
    for batch in tqdm.tqdm(dataset):
        # The final batch has been padded with fake examples so that the batch size is
        # the same as all other batches. The mask tells us which examples are fake.
        done_batches = which_batches_are_done(save_dir)
        if i > end_b:
            break
        if i < start_b: 
            i += 1
            continue
        if i in done_batches:
            i += 1
            logging.info(f'skipping batch_idx {i} since it exists already')
            continue
        mask = batch['__valid__']
        logits, _, indices_distr = model.apply({'params': checkpoint}, batch['image'])
    
        log_p = jax.nn.log_softmax(logits)
        preds = jnp.argmax(log_p, axis=1)
        true_lbl = jnp.argmax(batch['labels'], axis=1)

        process_indices(indices_distr, mask=mask, batch_idx=i, class_lbl=true_lbl, fdir=save_dir)

        ncorrect += jnp.sum((preds == true_lbl) * mask)
        ntotal += jnp.sum(mask)
        if i % 10 == 0:
          logging.info(f'Test accuracy, iteration {i}: {ncorrect / ntotal * 100:.2f}%')
        i += 1
    print(f'Test accuracy: {ncorrect / ntotal * 100:.2f}%')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_or_test', type=str, required=True)
    parser.add_argument('-s', '--save_dir', type=str, required=True)
    parser.add_argument('-b', '--start_b', type=int)
    parser.add_argument('-e', '--end_b', type=int)  # inclusive of both ends
    args = parser.parse_args()

    logging.basicConfig(filename=f'vmoe_datagen_{args.train_or_test}.log', 
                        level=logging.INFO, 
                        format='%(levelname)s: %(asctime)s %(message)s')

    model_config = get_config()
    model_cls = getattr(models, model_config.model.name)
    model = model_cls(deterministic=True, **model_config.model)

    # using this model: 'gs://vmoe_checkpoints/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012'
    checkpoint_prefix = 'vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012'
    checkpoint = partitioned.restore_checkpoint(prefix=checkpoint_prefix, tree=None)

    dataset_config_test = model_config.dataset.test
    dataset_test = input_pipeline.get_dataset(
        variant='test',
        name=dataset_config_test.name, 
        split=dataset_config_test.split, 
        batch_size=dataset_config_test.batch_size, 
        process=dataset_config_test.process
    )

    dataset_config_train = model_config.dataset.train
    dataset_train = input_pipeline.get_dataset(
        variant='train',
        name=dataset_config_train.name, 
        split=dataset_config_train.split, 
        batch_size=dataset_config_train.batch_size, 
        process=dataset_config_train.process
    )
    dataset = None
    if args.train_or_test == "train":
        dataset = dataset_train
    elif args.train_or_test == "test":
        dataset = dataset_test

    logging.info(f"start of generation file for save_dir = {args.save_dir}, BATCH_SIZE = {BATCH_SIZE}, data = {args.train_or_test}, from {args.start_b} to {args.end_b}")
    gen_data(model, dataset, checkpoint, save_dir=args.save_dir, start_b=args.start_b, end_b=args.end_b)
             #save_dir='/home/zl310/cs585_project/vmoe/expert_assign_test_ImageNetData')
