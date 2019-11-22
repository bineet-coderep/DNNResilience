# Resilience of a Deep Neural Network
COMP 562 Machine Learning - Final Project

The implementation is based on the paper ["Maximum Resilience of Artificial Neural Networks", Chih-Hong Cheng, Georg Nuhrenberg, and Harald Ruess](<https://arxiv.org/abs/1705.01040>).

## Description

This provides a list of APIs to compute the maximum amount of perturbation a Deep Neural Network can tolerate without a mis-classification. Using these API an implementation that computes the maximum amount of perturbation a Deep Neural Network can tolerate without a mis-classification is provided.

The main Perturbation Computation Engine can be found in the folder `DNN` 

## Dependencies

This code has following dependencies:

* Python Numpy
* Python [Keras](https://keras.io/)
* [Gurobi](https://www.gurobi.com/) Python Interface

## Install Dependecies

### Gurobi

- Please obtain appropriate Gurobi License from [here](http://www.gurobi.com/downloads/licenses/license-center). Please note that if you are using Academic License, you **should be in your University network** (VPN should work fine too) while installing the license. Please refer to this [link](https://www.gurobi.com/documentation/8.1/quickstart_windows/academic_validation.html) for details. After the license is installed properly, Gurobi can be used from home network.
- Install Gurobi. Please note that we will need Gurobi Python Interface. On-line documentation on installation can be found [here](http://www.gurobi.com/documentation/).
- Gurobi Python Interface can also be installed through [Anaconda](https://www.anaconda.com/). Details on installing Gurobi Python Interface through `conda` can be found [here](https://www.gurobi.com/documentation/8.1/quickstart_mac/installing_the_anaconda_py.html#section:Anaconda).

### Python Keras

Please see this [link](https://github.com/hsekia/learning-keras/wiki/How-to-install-Keras-to-Ubuntu-18.04) for installing Python Keras to Ubuntu 18.04.

## Bug Report

Please send an email to Bineet Ghosh at bineet@cs.unc.edu to report any bug.

