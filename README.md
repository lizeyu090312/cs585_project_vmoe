# Side-Channel Attacks on Mixture of Experts (MoE) models

This repository explores side-channel class inference attacks on MoE models for vision classification.
Specifically, we use the model developed in [Scaling Vision with Sparse Mixture of Experts](https://arxiv.org/abs/2106.05974)
as the model that we attack.

# Goal

We consider a distributed form of deploying MoE models where experts are deployed on different secure environments (e.g., GPU TEEs). 
In MoE layers, images are divided into patches and are then assigned to experts by the routing layer. Assuming the attacker has access over 
unencrypted communication channels between routing layers and MoE layers, the unencrypted communication allow
attackers to directly parse messages and extract the expert assignment of each patch. We use this patch-wise 
assignment to information about each image. \
In the more realistic scenario where communication between the routing layer and experts are encrypted, 
we use the communication size side-channel to infer the majority class in an input batch.

# Generating Data and Implementing the Attack

`data_gen_encrypted_train_wrap.py` collects data on the communication size side-channel using the encrypted communication scenario. \
The `model_encrypted/` directory contains code that trains and evaluates a light-weight attack model for inferring the majority class. 

`data_gen_unencrypted.py` collects data on fine-grained patch-wise expert routing information in the unencrypted communication scenario. \
The `model_unencrypted/` directory uses a ResNet to train the attack model for inferring the class of a input image given its 
expert routing assignments. 

# Project Writeup
Please see `CS585_Writeup.pdf` for our writeup. 