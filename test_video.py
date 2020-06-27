import os
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
from keras.models import load_model

# denseDepth specific
from layers import BilinearUpSampling2D
from utils import predict, display_images


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
print('Loading model...')
model = load_model('nyu.h5', custom_objects=custom_objects, compile=False)

def preprocess_img_new(imgsrc):
    img = Image.fromarray(imgsrc).resize((640, 480))
    img = np.asarray(img, dtype=float)[:, :, 0:3]
    x = np.clip(img / 255, 0, 1)

    inputs = np.stack([x], axis=0)
    return inputs


def get_depth(cap):
    inputs = preprocess_img_new(cap)
    outputs = predict(model, inputs)
    return outputs.copy()

def grab_frame(cap):
    ret,frame = cap.read()
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


# cap1 = cv2.VideoCapture("http://192.168.1.4:8080/video/mjpeg")
cap1 = cv2.VideoCapture(0)

resCap = grab_frame(cap1)


fig=plt.figure(figsize=(30,10))
columns = 3
rows = 1

fig.add_subplot(rows, columns, 1)
im1 = plt.imshow(resCap)

plt.xticks([])
plt.yticks([])

fig.add_subplot(rows, columns, 2)
# im2 = plt.imshow(resCap)
im2 = plt.imshow(display_images(get_depth(resCap)))

plt.xticks([])
plt.yticks([])

fig.add_subplot(rows, columns, 3)
im3 = plt.imshow(resCap)

plt.xticks([])
plt.yticks([])

plt.ion()

while True:
    resCap = grab_frame(cap1)
    im1.set_data(resCap)
    # im2.set_data(resCap)
    im2.set_data(display_images(get_depth(resCap)))
    im3.set_data(resCap)

    plt.pause(0.2)

plt.ioff() # due to infinite loop, this gets never called.
plt.show()
