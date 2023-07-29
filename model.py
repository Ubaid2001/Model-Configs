# import tensorflow as tf
# import cv2
# import numpy as np
# import os
# import keras
# from keras.preprocessing.Image import ImageDataGenerator
# from keras.layers import Dense, Flatten, Conv2D, Activation, Dropout
# from keras import backend as K
# from keras.models import Sequential, Model
# from keras.models import load_model
# from keras.optimizers import SGD
# from keras.callbacks import EarlyStopping,ModelCheckpoint
# from keras.layers import MaxPool2D
# # from google.colab.patches import cv2_imshow

import tensorflow as tf
from js2py import eval_js
from keras.callbacks import TensorBoard as tb
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Activation, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import os
import dlib
import cv2
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import datetime
import plotly.express as px
from random import randint
from IPython.display import display, Javascript
import js2py
from base64 import b64decode
from logdir import LogDir


# train_datagen = ImageDataGenerator(zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15)
# test_datagen = ImageDataGenerator()
#
# train_generator = train_datagen.flow_from_directory(“/content/drive/MyDrive/Rohini_Capstone/Car Images/Train Images”,target_size=(224, 224),batch_size=32,shuffle=True,class_mode=’categorical’)

data = open("processedData/wiki_data.pickle", "rb")
data = pickle.load(data)

datagen = ImageDataGenerator(zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15, horizontal_flip=True)

images = np.array(data['images'])
genders = np.array(data['gender'])
print('''The shape of the images array is : {}\n
The shape is an image is : {}\n
The shape of the gender array is : {}'''.format(images.shape, images[0].shape, genders.shape))

# print(f'1 row: {data["images"][-1]}')

# Find a way to auguemnt the data since the ImageDataGenerator.flow method is not working, as it needs a rank 4 array.
# g = 0
# f = 0
# info = []
# for d in genders:
#     de = d
#     if de == 0.0:
#         des = data["images"][g]
#         info.append(des)
#         # datagen.flow(data["images"][g], data["gender"][g], batch_size=1)
#         f += 1
#     else:
#         des = None
#     g += 1
#
# print(f)
# print(g)
# print(len(info))
#
# info = np.array(info)
#
# info.reshape(9431, 32, 32, 3)
#
# datagen.flow(info, None, batch_size=1)
# info = np.array(info)
#
# k = info[4].reshape(1, 32, 32, 3)
#
# l = datagen.flow(k, None, batch_size=1)
# plt.imshow(l[0])
# plt.show()

#
# print(len(info))
#
# plt.imshow()
# plt.show()

print(len(data["images"]))
print(len(data["gender"]))

f = 0
m = 0
for z in genders:
    if z == 0.0:
        f += 1
    else:
        # print("")
        m += 1

print(f)
print(m)


# plt.figure()
# i = 1
# while (i <= 4):
#     index = randint(0, len(images))
#     img = images[index]
#
#     plt.subplot(1, 4, i)
#     plt.imshow(img)
#     plt.title(genders[index][0])
#     plt.show()
#     i += 1

# gender_plotting = []
#
# for i in genders:
#     if i[0] == 1:
#         gender_plotting.append('Male')
#     else:
#         gender_plotting.append('Female')
#
# dataframe = pd.DataFrame({'gender' : gender_plotting})
# fig = px.histogram(dataframe, x="gender")
# fig.show()
# del dataframe
# del gender_plotting

genderCategorical = []

for i in genders:
    if i[0] == 1:
        genderCategorical.append([1.0, 0.0])
    else:
        genderCategorical.append([0.0, 1.0])
genderCategorical = np.array(genderCategorical)

train_images, test_images, train_gender, test_gender = train_test_split(images, genderCategorical,
                                                                        test_size=.2, shuffle=True, random_state=10)

if not os.path.exists("models/model.json"):
    layers = [Input(shape=(32,32,3))]
    no_of_conv_layers = (16,32, 64,128)

    for i in no_of_conv_layers:
        layers += [
                Conv2D(i, padding='same', kernel_size=(2,2)),
                   Activation('relu'),
                   BatchNormalization(),
                   MaxPooling2D(pool_size=(2,2), strides=2)
        ]

    layers += [
               Flatten(),
               Dense(512),
               Activation('relu'),
               BatchNormalization(),
               Dropout(0.25),
               Dense(128),
               Activation('relu'),
               BatchNormalization(),
               Dropout(0.25),
               Dense(64),
               Activation('relu'),
               BatchNormalization(),
               Dense(16),
               Activation('relu'),
               Dense(2),
               Activation('softmax')
    ]

    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # log_dir = datetime.datetime.now().strftime("%Y%m%D-%H%M%S")
    # log_dir = "logs/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [tb(log_dir="logs")]

    # train_images, test_images, train_gender, test_gender = train_test_split(images, genderCategorical,
    #                                                     test_size = .2, shuffle = True, random_state = 10)

    num_train_examples = len(train_images) * 0.8
    BATCH_SIZE = 64
    model.fit(train_images, train_gender, epochs = 10, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE),
              batch_size = BATCH_SIZE, shuffle=True, validation_split = 0.2,
              callbacks = callbacks)

    # print("\nEvaluating the Model\n")
    # model.evaluate(test_images, test_gender, callbacks = callbacks)

    # print(f'tensorboard url: {tb}')

    model_json = model.to_json()
    with open("models/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
else:
    json_file = open('models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    model = loaded_model
    print("Loaded model from disk")
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # log_dir = datetime.datetime.now().strftime("%Y%m%D-%H%M%S")
    # log_dir = "logs/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [tb(log_dir="logs")]

print("\nEvaluating the Model\n")
model.evaluate(test_images, test_gender, callbacks = callbacks)

def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )

# def take_photo( img_width = 48, img_height = 48, quality=0.8 ):
#   js = Javascript('''
#     async function takePhoto(img_width, img_height, quality) {
#       const div = document.createElement('div');
#       const capture = document.createElement('button');
#       capture.textContent = 'Capture';
#       div.appendChild(capture);
#
#       const video = document.createElement('video');
#       video.style.display = 'block';
#       const stream = await  navigator.mediaDevices.getUserMedia({video: {height:img_height, width:img_width}});
#
#       document.body.appendChild(div);
#       div.appendChild(video);
#       video.srcObject = stream;
#       await video.play();
#
#       // Resize the output to fit the video element.
#       google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
#
#       // Wait for Capture to be clicked.
#       await new Promise((resolve) => capture.onclick = resolve);
#       const canvas = document.createElement('canvas');
#       canvas.width = video.videoWidth;
#       canvas.height = video.videoHeight;
#       ctx = canvas.getContext('2d')
#       ctx.drawImage(video, 0, 0);
#       imageData  = ctx.getImageData(1,1, Math.round(img_width), Math.round(img_height))
#       stream.getVideoTracks()[0].stop();
#       div.remove();
#       return imageData.data
#     }
#     ''')
#   display(js)
#   data = eval_js('takePhoto({},{},{})'.format(img_width, img_height, quality))
#   return data

def take_photo():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            return frame

    cam.release()

    cv2.destroyAllWindows()

img = take_photo()

print(img)
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(image_rgb)
# plt.show()


img_height = 256
img_width = 256

# try:
#     # data = take_photo(img_height, img_width, 0.3)
#     img = []
#     data = take_photo()
#     for i in sorted(data, key = lambda x : int(x)):
#         img.append(data[i])
#     img = np.reshape(img, (img_height, img_width,4))
# except Exception as err:
#   # Errors will be thrown if the user does not have a webcam or if they do not
#   # grant the page permission to access it.
#   print(str(err))

# img = rgba2rgb(img)
# plt.imshow(img)
#
def extract_faces(img):
    cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
    faces_cnn = cnn_face_detector(img, 1)
    faces = []
    if(len(faces_cnn) > 0):
        for face in faces_cnn:
            offset_x , offset_y  = max(face.rect.left(),0),max(face.rect.top(),0)
            target_width, target_height = face.rect.right() - offset_x, face.rect.bottom() - offset_y

            target_width = min(target_width, img.shape[1]-offset_x)
            target_height = min(target_height, img.shape[0]-offset_y)
            print("face", offset_x,offset_y,target_width,target_height)

            face_img = tf.image.crop_to_bounding_box(img,
                                            offset_y, offset_x,
                                            target_height,target_width)

            face_img = tf.image.resize(face_img, (32, 32), method=tf.image.ResizeMethod.BICUBIC, antialias=True)
            face_img = tf.dtypes.cast(face_img, tf.int32)
            faces.append({'face' : face_img.numpy(), 'coordinates' : [offset_x, offset_y, target_width, target_height]})
    return faces

def show_output(img, faces, predictions):
    for i, data in enumerate(faces):
        coordinates = data['coordinates']
        x1 = coordinates[0]
        y1 = coordinates[1]
        x2 = coordinates[2] + x1
        y2 = coordinates[3] + y1
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)
        gender = 'Male' if np.argmax(predictions[i]) == 0 else 'Female'
        cv2.putText(img, gender, (x1-3, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 1)
    plt.imshow(img)
    plt.show()

# faces = extract_faces(image_rgb)
# print(faces)
# predict_X = []
# for face in faces:
#     predict_X.append(face['face'])
#
# predict_X = np.array(predict_X)
#
# predictions = []
# if predict_X.shape[0] > 0:
#     predictions = model.predict(predict_X)
#
# show_output(image_rgb, faces, predictions)