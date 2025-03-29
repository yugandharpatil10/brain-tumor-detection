import cv2
from PIL import Image
import numpy as np
from keras.models import load_model

model = load_model('BrainTumor10Epochs.h5')

image = cv2.imread('D:\\Project\\Brain Tumour Detection\\pred\\pred0.jpg')

img = Image.fromarray(image)

img = img.resize((64, 64))

img = np.array(img)

input_img = np.expand_dims(img, axis=0)

# print(img)

result = model.predict(input_img)

print(result)

if result[0] > 0.5:
    print("Tumor Detected")
else:
    print("No Tumor Detected")