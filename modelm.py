import random
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
import albumentations as A

data = open("processedData/wiki_data.pickle", "rb")
data = pickle.load(data)

datagen = ImageDataGenerator(zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15, horizontal_flip=True)

images = np.array(data['images'])
genders = np.array(data['gender'])
print('''The shape of the images array is : {}\n
The shape is an image is : {}\n
The shape of the gender array is : {}'''.format(images.shape, images[0].shape, genders.shape))


f = 0
m = 0
index = 0
mod_genders = []
mod_images = []
for z in genders:
    if z == 0.0:
        mod_images.append(images[index])
        mod_genders.append(genders[index])
        f += 1
    elif m <= f:
        # print("")
        mod_images.append(images[index])
        mod_genders.append(genders[index])
        m += 1

    index += 1

mod_images = np.array(mod_images)
mod_genders = np.array(mod_genders)

print(f'This is the number of females: {f}')
print(f'This is the number of males: {m}')

print(f'This is the length of modded genders var: {len(mod_genders)}')
print(f'This is the length of modded images var: {len(mod_images)}')

print('''The shape of the images array is : {}\n
The shape is an image is : {}\n
The shape of the gender array is : {}'''.format(mod_images.shape, mod_images[0].shape, mod_genders.shape))

genderCategorical = []

for i in mod_genders:
    if i[0] == 1:
        genderCategorical.append([1.0, 0.0])
    else:
        genderCategorical.append([0.0, 1.0])
genderCategorical = np.array(genderCategorical)

train_images, val_images, train_gender, val_gender = train_test_split(mod_images, genderCategorical,
                                                                        test_size=.2, shuffle=True, random_state=10)

train_images, test_images, train_gender, test_gender = train_test_split(train_images, train_gender,
                                                                        test_size=.2, shuffle=True, random_state=10)



# datagen.fit(train_images)
# for X_batch, y_batch in datagen.flow(train_images, train_gender, batch_size=9):
#     for i in range(0, 9):
#         plt.imshow(X_batch[i])
#         plt.show()
#     break

# plt.imshow(train_images[0])
# plt.show()

# def visualize(image):
#     plt.figure(figsize=(10, 10))
#     plt.axis('off')
#     plt.imshow(image)
#     plt.show()
#
# image = train_images[0]
# # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# transform = A.Compose(
#     [A.CLAHE(),
#      A.RandomRotate90(),
#      A.Transpose(),
#      A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50,
#                         rotate_limit=45, p=.75),
#      A.Blur(blur_limit=3),
#      A.OpticalDistortion(),
#      A.GridDistortion(),
#      A.HueSaturationValue()])
#
# augmented_image = transform(image=image)['image']
# visualize(augmented_image)

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
callbacks = [tb(log_dir="logs")]


num_train_examples = len(train_images) * 0.8
BATCH_SIZE = 64

# model.fit(train_images, train_gender, epochs = 10, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE),
#               batch_size = BATCH_SIZE, shuffle=True, validation_split = 0.3,
#               callbacks = callbacks)
#  77% accuracy and 0.46 loss

# model.fit(datagen.flow(train_images, train_gender, batch_size=BATCH_SIZE),
#                     epochs=10,
#                     steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE),
#                     shuffle=True, validation_data=(val_images, val_gender),
#                     callbacks=callbacks)
# loss: 0.3882 - accuracy: 0.8171

model.fit(datagen.flow(train_images, train_gender, batch_size=BATCH_SIZE),
                    epochs=15,
                    steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE),
                    shuffle=True, validation_data=(val_images, val_gender),
                    callbacks=callbacks)
# loss: 0.3460 - accuracy: 0.8419


print("\nEvaluating the Model\n")
model.evaluate(test_images, test_gender, callbacks=callbacks)

model.summary()



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

# il = os.listdir("./imdb_mat_folder-images/00")
# il = np.array(il)
# # print(il[0])
#
# ran_img = cv2.imread("./imdb_mat_folder-images/00/" + il[random.choice(range(0, len(il)))])
# ran_img = cv2.cvtColor(ran_img, cv2.COLOR_BGR2RGB)
# # plt.imshow(ran_img)
# # plt.show()
# # print(ran_img)
#
# faces = extract_faces(ran_img)
# # print(faces)
#
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
# show_output(ran_img, faces, predictions)



# img = take_photo()
# print(img)
# image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
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

