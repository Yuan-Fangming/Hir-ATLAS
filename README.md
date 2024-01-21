# Local positional graphs and attentive local features for a data and runtime-efficient hierarchical place recognition pipeline



![aaa](/images/Zeichnung_v2.svg)
This repository contains the source code for the accepted  IEEE Robotics and Automation Letters(RA-L) paper: Local positional graphs and attentive local features for a data and runtime-efficient hierarchical place recognition pipeline.
The paper can be found in ....$link$....

The pipeline is a hybrid implementation of Python and C/C++, where Python is used as a glue language to integrate software packages for feature extraction, database preparation, and the LPG algorithm.  C/C++ is used to integrate Cublas running on GPU and CPU-based multi-thread to accelerate software packages such as feature mutual comparison, RANSAC, and LPG algorithm.
