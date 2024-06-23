import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report

model=load_model('BrainTumor10epochscatagaries.h5')

image=cv2.imread(r"C:\Users\hp\Desktop\BrainTumor_Classification_DL\uploads\pred7.jpg")

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)
# print(input_img)
result=model.predict(input_img)
print(result)
fin_res = print(np.argmax(result))
# np.where(fin_res>0, print)
