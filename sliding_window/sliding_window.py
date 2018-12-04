import base64, sys, json
import tensorflow as tf
import os



PROJECT = 'sidewalk-dl'
BUCKET = 'sidewalk_crops_subset'
REGION = 'us-central1'

MODEL = 'sidewalk'
VERSION = 'resnet'

def predict_json(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    service = googleapiclient.discovery.build('ml', 'v1')
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
with open('null14.jpg', 'r') as ifp, open('test.json', 'w') as ofp:
    image_data = ifp.read()
    img = base64.b64encode(image_data)
    json.dump({"image_bytes": {"b64": img}}, ofp)

#predictions = predict_json(PROJECT, MODEL, xx, VERSION)
#print predictions
