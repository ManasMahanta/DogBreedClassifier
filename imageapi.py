from keras.utils.generic_utils import CustomObjectScope
import json
import keras
import cv2

from keras.models import load_model
from PIL import Image
import numpy as np
from flasgger import Swagger
import numpy as np

from flask import Flask, request
app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    """Example endpoint returning a prediction of mnist
    ---
    parameters:
        - name: image
          in: formData
          type: file
          required: true
    """
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model('./dog_breed_mobileNet.h5')
    with open('labels.json', 'r') as fp:
        labels = dict(json.load(fp))
    im = Image.open(request.files['image'])
    #print ('hello')

    im2arr = np.array(im)
    #input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    input_im = cv2.resize(im2arr, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3)

    # Get Prediction
    #print(model.predict(input_im, 1, verbose = 0))
    pred = np.argmax(model.predict(input_im, 1, verbose = 0), axis=1)
    index=str(list(pred)[0])
    print(labels[index])

    return str(labels[index])

if __name__ == '__main__':
    app.run()