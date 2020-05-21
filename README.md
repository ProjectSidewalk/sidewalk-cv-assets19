# FAQ

## Can we use your dataset?

Our data is most certainly available, with some caveats. Unfortunately, due to licensing restrictions with Google, we can't share the actual crops from Google Streetview imagery, but we can share the X,Y coordinates of our labels, as well as the panorama ID.

Then, using code we provide, you can download the panorama from Google's servers, and make crops from it.

CSVs containing labels this information are [here](https://github.com/ProjectSidewalk/sidewalk-cv-assets19/tree/master/dataset_csvs).

The code to download panoramas and crop them is [here](https://github.com/ProjectSidewalk/sidewalk-panorama-tools). You want the "DownloadRunner" and "CropRunner" scripts, respectively.

Read on for more information...


# Overview

This repository provides tools to train a neural network to detect sidewalk features in Google Streetview imagery, and tools to use a trained network. Everything is implemented in Python and Pytorch. For the purposes of our [2019 ASSETS submission](https://drive.google.com/file/d/1spidyhQpg-_FjRwTUeC_QSH1F1-AojPE/view?usp=sharing), which you might want to read, the sidewalk features we focus on detecting are:
- Curb Ramp
- Missing Curb
- Surface Problem
- Obstruction

We add a fifth feature, **null**, to these categories to enable the network to detect the *absence* of sidewalk features.

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
Once this is done, activate the environment with:
```
conda activate sidewalk_pytorch
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

### Setup

As mentioned above, if you're planning on using a model that requires additional features, you should use the `TwoFileFolder` dataloader provided by  `pytorch_pretrained/TwoFileFolder.py`.
The dataloader expects your files to be organized with the following directory structure:
```
root/
     label1/
            file1.jpg
            file1.json
            file2.jpg
            file2.json
     label2/
            file3.jpg
            file3.json
            file4.jpg
            file4.json
```

Where the `.jpg` files are square crops (of any resolution) and the `.json` files contain the following fields:
```
{"dist to cbd": 4.094305012075221, "bearing to cbd": 64.74029765051874, "crop size": 1492.1348109969322, "sv_x": 9300.0, "sv_y": -1500.0, "longitude": -76.967779, "pano id": "__1c3_5IArbrml1--v7meQ", "dist to intersection": 23.45792342820621, "block middleness": 42.95327159437733, "latitude": 38.872448, "pano yaw": -179.67633056640602, "crop_y": 4828.0, "crop_x": 2655.9685763888974}
```

These crops and `.json` files can be produced easily and simulataneously using the `bulk_extract_crops` function from `GSVutils/utils.py`.
This function takes as input a `.csv` with the following columns:
```
Pano ID, SV_x, SV_y, Label, Photog Heading, Heading, Label ID 
```

### Using the Model

Once you've got your crops and additional feature `.json` files, you're ready to go. First, define your data transforms, and build the dataset from your crop directory using `TwoFileFolder`:

```python
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# the dataset loads the files into pytorch vectors
image_dataset = TwoFileFolder(dir_containing_crops, meta_to_tensor_version=2, transform=data_transform)

# the dataloader takes these vectors and batches them together for parallelization, increasing performance
dataloader    = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=4)

# this is the number of additional features provided by the dataset
len_ex_feats = image_dataset.len_ex_feats
dataset_size = len(image_dataset)
```

With this done, we can load the model itself. First, we load the architecture, then we load the weights from the `.pt` file onto the architecture:

```python
model_ft = extended_resnet18(len_ex_feats=len_ex_feats)

try:
    model_ft.load_state_dict(torch.load(model_path))
except RuntimeError as e:
    model_ft.load_state_dict(torch.load(model_path, map_location='cpu'))
model_ft = model_ft.to( device )
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# this tells pytorch to not change the weights, since we're using the model to get predictions, not training
model_ft.eval()
```

Now, we can actually compute the predictions. We do this by looping over all the data in the dataloader, and computing the predictions. We accumulate these predictions in `pred_out`, and for simplicity, we accumulate the paths in `paths_out`:

```python
for inputs, labels, paths in dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    # zero the parameter gradients
    optimizer_ft.zero_grad()

    with torch.set_grad_enabled(False):
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)

        paths_out += list(paths)
        pred_out  += list(outputs.tolist())
```

With this finished, we now have our predictions (in integer form), and the corresponding image paths in `paths_out`. For example, if `paths_out[0]`is `/example_dir/example_label/example_img.jpg`, then `preds_out[0]` will be the integer prediction for `example_img.jpg`. What do I mean by integer prediction? To save memory, PyTorch assigns each string label an integer index, and stores those indices instead of the strings. Our labels are `('Missing Cut', "Null", 'Obstruction', "Curb Cut", "Sfc Problem")`, so if `paths_out[0]` is 0, then the model assigned a prediction of `Missing Curb Ramp` to the image `example_img.jpg`.

Now we're pretty much finished! We can wrap this all into a single easy function for you to use for whatever purpose you like. This returns a list of (img_path, predicted_label) tuples.:

```python
def predict_from_crops(dir_containing_crops, model_path):
    ''' use the TwoFileFolder dataloader to load images and feed them
        through the model
        a list of (img_path, predicted_label) tuples
    '''
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    print "Building dataset and loading model..."
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_dataset = TwoFileFolder(dir_containing_crops, meta_to_tensor_version=2, transform=data_transform)
    dataloader    = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=4)

    len_ex_feats = image_dataset.len_ex_feats
    dataset_size = len(image_dataset)

    panos = image_dataset.classes

    print("Using dataloader that supplies {} extra features.".format(len_ex_feats))
    print("")
    print("Finished loading data. Got crops from {} panos.".format(len(panos)))


    model_ft = extended_resnet18(len_ex_feats=len_ex_feats)

    try:
        model_ft.load_state_dict(torch.load(model_path))
    except RuntimeError as e:
        model_ft.load_state_dict(torch.load(model_path, map_location='cpu'))
    model_ft = model_ft.to( device )
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    model_ft.eval()

    paths_out = []
    pred_out  = []

    print "Computing predictions...."
    for inputs, labels, paths in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer_ft.zero_grad()

        with torch.set_grad_enabled(False):
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)

            paths_out += list(paths)
            pred_out  += list(outputs.tolist())

    print "Finished!"
    pytorch_label_from_int = ('Missing Cut', "Null", 'Obstruction', "Curb Cut", "Sfc Problem")
    str_predictions = [pytorch_label_from_int[np.argmax(x)] for x in pred_out]

    return zip(paths_out, str_predictions)
```

## Using a Model for Labeling

Todo
