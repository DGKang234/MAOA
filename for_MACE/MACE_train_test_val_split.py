import sys
import os
import subprocess
import numpy as np
from ase import io

# required module for MACE training

#module load amd-modules
#module load Python/3.10.4-GCCcore-11.3.0
#module load  foss/2022a CUDA/11.7.1  NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.1
#source ~/venv/mace/bin/activate

training_data_path = sys.argv[1]

#orig_train_file = io.read(training_data_path, index=":")
#id_1 = int(len(orig_train_file)*0.8)
#id_2 = int(len(orig_train_file)*0.9)
#train, test, valid = np.split(orig_train_file, [id_1, id_2])
#
#train = io.read(training_data_path, index=f":{len(train)}")
#test = io.read(training_data_path, index=f"{len(train)}:{len(train)+len(test)}")
#valid = io.read(training_data_path, index=f"{len(train)+len(test)}:")
#
#train = io.write('./Training_set_test.xyz', train, format='extxyz')
#test = io.write( './Testing_set_test.xyz', test, format='extxyz')
#valid = io.write('./Validation_set_test.xyz', valid, format='extxyz')
##test_run_sample = io.write('./FIT/tesing_sample.xyz', sample, format='extxyz')

orig_train_file = io.read(training_data_path, index=":")

train = []
valid = []
test = []

for i, data in enumerate(orig_train_file):
    if i % 5 < 3:  # First three items in every set of five
        train.append(data)
    elif i % 5 == 3:  # Fourth item in every set of five
        valid.append(data)
    elif i % 5 == 4:  # Fifth item in every set of five
        test.append(data)

# Write the data to the respective files
io.write('./Training_set_test.xyz', train, format='extxyz')
io.write('./Validation_set_test.xyz', valid, format='extxyz')
io.write('./Testing_set_test.xyz', test, format='extxyz')


