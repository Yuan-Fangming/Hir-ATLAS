# Local positional graphs and attentive local features for a data and runtime-efficient hierarchical place recognition pipeline


## Introduction

This repository contains the source code for the accepted  IEEE Robotics and Automation Letters(RA-L) paper: Local positional graphs and attentive local features for a data and runtime-efficient hierarchical place recognition pipeline.
The paper can be found in ....$link$....

The pipeline is a hybrid implementation of Python and C/C++, where Python is used as a glue language to integrate software packages for feature extraction, database preparation, and the LPG algorithm.  C/C++ is used to integrate Cublas running on GPU and CPU-based multi-thread to accelerate software packages such as feature mutual comparison, RANSAC, and LPG algorithm.

## Requirments 
### Python:
`scipy-1.10.1  h5py-3.10.0 hdf5storage-0.1.18  numpy-1.23.5 matplotlib-3.7.4`
### Nvidia tool chain:
`nvcc-12.1`

## Compile the acceleration package
First cd to AccLib directory
`cd AccLib/`


Compiling `AccLibcu/VIPRMatAcc.cpp` for GPU accelerated feature comparison, where `$NV-tool-chain-path$` is the path to the nvcc e.g /usr/local/cuda-12.1/bin

`$NV-tool-chain-path$/nvcc --device-debug --debug -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -Xcompiler -fPIC -ccbin g++ -c -o "cuVIPRMatAcc.o" "cuVIPRMatAcc.cpp"`

`$NV-tool-chain-path$/nvcc --cudart=static -L/usr/local/cuda-12.1/targets/x86_64-linux/lib -ccbin g++ --shared -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o "libcuVIPRACCLib"  ./cuVIPRMatAcc.o   -lcublas -lcublasLt`

Compiling `AccLib/vipracc.cpp` for multi-thread LPG, MM, and RANSAC algorithms

`g++ -O3 -Ofast -g3 -Wall -c -fmessage-length=0 -fPIC -MMD -MP -MF"vipracc.d" -MT"vipracc.o" -o "vipracc.o" "vipracc.cpp"`

`g++ -L/usr/local/lib -shared -pthread -o "libvipracc"  vipracc.o   -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_features2d -lopencv_calib3d -lopencv_flann`


# The source code is still under preparation for a compact software package based on the experiment-level code.
