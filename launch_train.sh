#!/bin/bash

#$-l rt_AG.small=1
#$-j train_calvin_residual
#$-cwd

source ~/.zshrc
mamba activate mdt_policy
python test_train.py
