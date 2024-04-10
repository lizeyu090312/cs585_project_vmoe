import jax
import numpy as np
from jax import numpy as jnp

import tensorflow as tf

from vmoe.nn import models
from vmoe.data import input_pipeline
from vmoe.checkpoints import partitioned

from vmoe.configs.vmoe_paper.vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012 import get_config, IMAGE_SIZE, BATCH_SIZE


model_config = get_config()
model_cls = getattr(models, model_config.model.name)
model = model_cls(deterministic=True, **model_config.model)

# using this model: 'gs://vmoe_checkpoints/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012'
# checkpoint_prefix = 'vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012'
# checkpoint = partitioned.restore_checkpoint(prefix=checkpoint_prefix, tree=None)

dataset_config_test = model_config.dataset.test
dataset_test = input_pipeline.get_dataset(
    variant='test',
    name=dataset_config_test.name, 
    split=dataset_config_test.split, 
    batch_size=dataset_config_test.batch_size, 
    process=dataset_config_test.process,
    shuffle_seed=20
)

dataset_config_train = model_config.dataset.train
dataset_train = input_pipeline.get_dataset(
    variant='train',
    name=dataset_config_train.name, 
    split=dataset_config_train.split, 
    batch_size=dataset_config_train.batch_size, 
    process=dataset_config_train.process
)

i = 0
for batch in dataset_train:
    mask = batch['__valid__']
    true_lbl = np.argmax(batch['labels'], axis=1)
    print(true_lbl[20:50])
    if i > 10:
        break
    i += 1
    