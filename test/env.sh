#!/bin/bash

module load apps/anaconda3/5.2.0
module rm   compiler/dtk/21.10
module load compiler/dtk/23.04
module load compiler/cmake/3.15.6

alias python=/public/software/apps/anaconda3/5.2.0/bin/python3

export LD_LIBRARY_PATH=../rocSOLVER-rocm-5.4.3/build/release/rocsolver-install/lib:${LD_LIBRARY_PATH}

echo $LD_LIBRARY_PATH

