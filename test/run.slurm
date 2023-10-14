#!/bin/bash

#SBATCH -J gesvd_test
#SBATCH -p wzhdnormal
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o job.out
#SBATCH -e job.err
#SBATCH --gres=dcu:4
#SBATCH --exclusive

#hostfile=./$SLURM_JOB_ID
#scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}
# #srun hostname |sort |uniq -c |awk '{printf "%s slots=%s\n",$2,$1}' > hostfile

hostname

module load apps/anaconda3/5.2.0
module rm   compiler/dtk/21.10
module load compiler/dtk/23.04
module load compiler/cmake/3.15.6

alias python=/public/software/apps/anaconda3/5.2.0/bin/python3

export LD_LIBRARY_PATH=../rocSOLVER-rocm-5.4.3/build/release/rocsolver-install/lib:${LD_LIBRARY_PATH}

#ldd -r gesvd_test

# ./gesvd_test m n func test iters
#     m:
#     n:
#  func:
#  test:
# iters:

m=$1
n=$2
func=$3
test=$4
iters=$5

./gesvd_test $m $n $func $test $iters

