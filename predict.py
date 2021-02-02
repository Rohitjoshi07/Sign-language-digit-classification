import cv2
import numpy as np
from keras.models import model_from_json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


'''
This program will predict the output only when you provide the path of the image.
Further development to predict live on web app is under progress.
'''

model_file = open('Data/Model/model.json', 'r')
model = model_file.read()
model_file.close()
model = model_from_json(model)
# Getting weights
model.load_weights("Data/Model/weights.h5")


def predict(model, img):
    y = model.predict(img)
    ans = np.argmax(y, axis=1)
    store = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine',
             0: 'Zero'}

    return store[ans[0]], y[0][ans][0]


def image_preprocess(pic):
    img_height, img_width = pic.shape[:2]
    side_width = int((img_width - img_height) / 2)
    pic = pic[0:img_height, side_width:side_width + img_height]
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    width = 64
    height = 64
    dim = (width, height)

    # resize image
    resized = cv2.resize(pic, dim, interpolation=cv2.INTER_AREA)

    #print('Resized Dimensions : ', resized.shape)

    img = 1 - np.array(resized).astype('float32') / 255.
    img = img.reshape(64, 64, 1)
    img =img[np.newaxis,:,:,:]
    return img



img_path = input("Enter path of your image: ")
pic = cv2.imread(img_path)
img=image_preprocess(pic)
Y_string, Y_possibility = predict(model, img)
print(f"Output:{Y_string}, Possibility: {round(float(Y_possibility),2)*100}%")
