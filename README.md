# Overview

This repository provides tools to train a neural network to detect sidewalk features in Google Streetview imagery, and tools to use a trained network. Everything is implemented in Python and Pytorch. For the purposes of our 2019 ASSETS submission, the sidewalk features we focus on detecting are:
- Curb Ramp
- Missing Curb
- Surface Problem
- Obstruction

We add a fifth feature, **Null**, to these categories to enable the network to detect the __absence__ of sidewalk features.

## Network Architecture

A significant point of the 2019 ASSETS paper focused on experimenting with different network architectures to improve performance. All our architectures are based upon Resnet, a popular family of neural network architectures that achieves state of the art performance on the ImageNet dataset.

The resnet architecture takes as input square color images, in the form of a 244 x 244 x 3 channel (RGB) vector. Instead of feeding an entire GSV panorama into the network, we input small crops from a panorama. We modify this network architecture by incorporating **additional features**, loosely divided into:
- **Positional Features**, which describe where in the panorama a (potential) label is located, such as the X and Y coordinates in the panorama image, the yaw degree, and the angle above/below the horizon.
- **Geographic Features**, which describe where in the city the panorama is located. These include the distance and compass heading from the panorama to the CBD, the position in the street block, and the distance to the nearest intersection.

## Use Cases

We developed the system with the intention of applying it to two different tasks. While there is much in common between our two approaches for these two tasks, there are some differences, which are important to be aware of.

## Validation Task

For validation, the neural network is input square crops taken from a GSV panorama, and attempts to identify the presence or absence of an accesiblity problem by classifying the image as a curb ramp, missing curb ramp, surface problem, or obstruction, or null. To achieve the best performance on this task, we trained the network on crops from GSV imagery which are directly centered around crowdsourced labels. To create examples of "null" crops, we randomly sampled crops from the imagery.

### Labeling Task

For labeling, the model is tasked with locating and labeling all of the accessibility problems in an entire GSV panorama. Our approach for this task uses a sliding window technique, a standard technique for object detection in the computer vision community, which breaks the large scene into small, overlapping crops that are then passed into a neural network for classification.
The neural network outputs a single predicted class for each crop: curb ramp, missing curb ramp, surface problem, obstruction, or null. Crops with a predicted class of null are ignored, and the remaining predictions are then clustered using non-maximum suppression. Overlapping predictions for a given label type are grouped together, and the prediction with the highest neural network output value or ‘strength’ is kept, while weaker predictions are suppressed.

# Setup

For development, we used Ananconda to manage all neccesary Python packages. The `pytorch_pretrained/environment.yml` file should make it easy to create a new conda environment with the neccesary packages installed. 

To do so, install anaconda, then `cd` into the `pytorch_pretrained` directory, and run:
```
conda env create -f environment.yml
```

# Training a Model

Todo

# Using a Model

This section assumes that you already have a trained model, and you would like to use this model to validate or label GSV imagery.
A large number of models are included in this repository, in the  `pytorch_pretrained/models` directory.
In this directory, each model is a `*.pt` file, which stores the parameters of the model which are then applied to the pre-defined architecture which is defined in `pytorch_pretrained/resnet_extended*.py`.
The various models that are in  `pytorch_pretrained/models` have been trained on a variety of different architectures incorporating different sets of the additional features described in the Overview, and trained on different datasets.

If the model you would like to use requires additional features, then you must use the `TwoFileFolder` dataloader, which makes it easy to load both a crop and its associated positional and geographic features into a single PyTorch vector.

## Using a Model for Validation

Pass

## Using a Model for Labeling

Todo