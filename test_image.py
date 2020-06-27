# I copied @yasserius-ml implementation and modified

import os
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from keras.models import load_model
from skimage import io
# denseDepth specific
from layers import BilinearUpSampling2D
from utils import predict, display_images


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
print('Loading model...')
model = load_model('nyu.h5', custom_objects=custom_objects, compile=False)

def preprocess_img(img_path):
    """
    The input image(s) is opened using PIL, the A-channel is removed (if it exists),
    and the pixel values are normalized into the range 0-1.

    INPUT:
    a list of image path(s).

    OUTPUT:
    a numpy array of the stacked images.
    """
    loaded_images = []
    for file1 in img_path:
        img = Image.open(file1).resize((640, 480))
        img = np.asarray(img, dtype=float)[:, :, 0:3]
        x = np.clip(img / 255, 0, 1)
        loaded_images.append(x)
    inputs = np.stack(loaded_images, axis=0)
    return inputs

def preprocess_img_new(imgsrc):
    img = Image.open(imgsrc).resize((640, 480))
    img = np.asarray(img, dtype=float)[:, :, 0:3]
    x = np.clip(img / 255, 0, 1)

    inputs = np.stack([x], axis=0)
    return inputs



inputs = preprocess_img(["tt1.jpg"])

print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)

#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')

# Display results
viz = display_images(outputs.copy(), inputs.copy())
plt.figure(figsize=(20,10))
plt.imshow(viz)
plt.xticks([])
plt.yticks([])
plt.savefig('test.png')
plt.show()
