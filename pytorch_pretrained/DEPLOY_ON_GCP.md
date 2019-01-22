# Notes on Training on GCP

## Creating a VM Instance

First, you'll need a Google Cloud Platform instance to run the code on. I deployed one from the Click to Deploy Cloud Marketplace. I used the Deep Learning VM instance, made sure I had a quota which permitted GPU usage, and made sure the Pytorch framework was selected from the dropdown. This will install (almost) all the dependencies for you.

## Configuring the VM Instance

Once you've SSH'd into your GCP instance, there are two tasks to complete:

### Cloning the Repo

First, clone the repo using `git clone <url_of_this_repo>`

### Install GCSFUSE

You'll need to install GCSFUSE in order to mount your Google Cloud Storage bucket to the VM filesystem.

Follow installation instructions from [here](https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/installing.md):

```
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update
sudo apt-get install gcsfuse
```

Then, create a directory such as `~/sidewalk_test_crops/`

and mount the cloud bucket with something like
```
gcsfuse --implicit-dirs sidewalk_test_crops sidewalk_test_crops
```

where the first argument is the bucket, and the second is the local location to mount to.

** note ** the use of the `--implicit-dirs` flag, which tells GCSFUSE to create impicit directories. For more details, see [here](https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/semantics.md#implicit-directories)

## Running the Training Code

Now, you're ready to run the training code. `cd` into the `pytorch_pretrained` directory, and modify `train.py` to use the appropriate model, dataset, and save location/filename.

Then, run the training program:
```
python train.py > train.log --nohup &
```

The `--nohup &` makes sure that the code continues to run even if your SSH session is disrupted. To follow training progress, run:
```
tail -f train.log
```