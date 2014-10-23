#!/bin/bash
# declare -x LD_LIBRARY_PATH=":/usr/local/cuda-5.5/lib64"
declare -x LD_LIBRARY_PATH=":/usr/local/cuda-6.0/lib64"
export | grep cuda
port=$(expr 40100 + $1)
cd $1
pwd
#read -p "Press any key to continue... " -n1 -s

# turn on debug mode
set -x

OPENBLAS_MAIN_FREE=1 THEANO_FLAGS=device=gpu$2,force_device=True,base_compiledir=./compile spearmint \
--driver=local --method=GPEIOptChooser --method-args=use_multiprocessing=0,noiseless=1 --polling-time=15 --max-concurrent=$3 -w --port=$port ./config.pb

# turn off debug mode
set +x

read -p "Press any key to continue... " -n1 -s
