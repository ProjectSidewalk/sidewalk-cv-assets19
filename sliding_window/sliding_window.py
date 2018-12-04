import base64, sys, json
import tensorflow as tf
import os
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery

# will need to install Google API client
# pip install --upgrade google-api-python-client

GCLOUD_SERVICE_KEY = '/home/gweld/gcloud_acct_key.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GCLOUD_SERVICE_KEY

PROJECT = 'sidewalk-dl'
BUCKET = 'sidewalk_crops_subset'
REGION = 'us-central1'

MODEL = 'sidewalk'
VERSION = 'resnet'

def predict_json(project, model, instances, version=None):
    # from:
    # https://cloud.google.com/ml-engine/docs/tensorflow/online-predict#creating_models_and_versions
    service = discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


#with tf.gfile.FastGFile('gs://sidewalk_crops_subset/img/2.jpg', 'r') as ifp:
# with open('null14.jpg', 'r') as ifp, open('test.json', 'w') as ofp:
#     image_data = ifp.read()
#     img = base64.b64encode(image_data)
#     json.dump({"image_bytes": {"b64": img}}, ofp)

def predict_label(img_path):
	with open(img_path, 'r') as imgfile:
		image_data = imgfile.read()
		img = base64.b64encode(image_data)
		instances = {'image_bytes' : {'b64': img}}
		predictions = predict_json(PROJECT, MODEL, instances, VERSION)
		print predictions

predict_label('38.jpg')

