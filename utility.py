import numpy as np
from keras.preprocessing import image
from keras.models import Model
import tensorflow as tf

def img_to_encoding(img_path, model):
    img = image.load_img(img_path, target_size=(96, 96))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.
    encoding = model.predict_on_batch(img)
    return encoding[0]

def resize_img(img_path):
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (96, 96))
    cv2.imwrite(img_path, img)
