import tensorflow as tf
from keras.callbacks import TensorBoard as tb
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Activation, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import sys
import cv2
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from datetime import datetime
import mtcnn.mtcnn

# This will remove the warnings from tensorflow about AVX
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print()
print(f"Python {sys.version}")
gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

global model
image = None

def config_model():
    # data = open("./processedData/new_wiki_data.pickle", "rb")
    # data = pickle.load(data)
    # open("./processedData/new_wiki_data.pickle", "rb").close()

    data = open("../../processedData/wiki_data.pickle", "rb")
    data = pickle.load(data)
    open("../../processedData/wiki_data.pickle", "rb").close()

    datagen = ImageDataGenerator(zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                 horizontal_flip=True)

    images = np.array(data['images'])
    genders = np.array(data['gender'])

    # This makes the amount of female and male in the dataset equal.

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

    del images, genders, data

    mod_images = np.array(mod_images)
    mod_genders = np.array(mod_genders)

    genderCategorical = []

    for i in mod_genders:
        if i[0] == 1:
            genderCategorical.append([1.0, 0.0])
        else:
            genderCategorical.append([0.0, 1.0])
    genderCategorical = np.array(genderCategorical)
    # print(len(mod_images))

    del mod_genders

    train_images, val_images, train_gender, val_gender = train_test_split(mod_images, genderCategorical,
                                                                          test_size=.2, shuffle=True,
                                                                          random_state=10)

    train_images, test_images, train_gender, test_gender = train_test_split(train_images, train_gender,
                                                                            test_size=.2, shuffle=True,
                                                                            random_state=10)

    del mod_images, genderCategorical

    # # This file has a model that achieves 87% accuracy.
    # modelFile = "models/new_model_vgg16(2).json"
    # fileJson = "models/new_model_vgg16(2).json"
    # fileH5 = "models/new_model_vgg16(2).h5"

    # This file has a model that achieves ..% accuracy.
    modelFile = "models/final_model_vgg16.json"
    fileJson = "models/final_model_vgg16.json"
    fileH5 = "models/final_model_vgg16.h5"

    # This file has a model that achieves 90% accuracy.
    # fileJson = "models/new_model_vgg16(3).json"
    # fileH5 = "models/new_model_vgg16(3).h5"
    # modelFile = "models/new_model_vgg16(3).json"

    # # This file has a model that achieves 91% accuracy.
    # modelFile = "models/new_model_vgg16(4).json"
    # fileJson = "models/new_model_vgg16(4).json"
    # fileH5 = "models/new_model_vgg16(4).h5"

    # # This file has a model that achieves 9121% accuracy.
    # modelFile = "models/new_model_vgg16(5).json"
    # fileJson = "models/new_model_vgg16(5).json"
    # fileH5 = "models/new_model_vgg16(5).h5"

    # # This file has a model that achieves ..% accuracy.
    # modelFile = "models/new_model_vgg16(6).json"
    # fileJson = "models/new_model_vgg16(6).json"
    # fileH5 = "models/new_model_vgg16(6).h5"

    if not os.path.exists(modelFile):

        FC = 2048
        E = 1
        fre = -20

        # base_model = tf.keras.applications.VGG16(input_shape=(128, 128, 3), include_top=False, weights="imagenet")
        base_model = tf.keras.applications.VGG16(input_shape=(32, 32, 3), include_top=False, weights="imagenet")

        for layer in base_model.layers[:fre]:
            layer.trainable = False

        # Building Model
        model = Sequential()
        model.add(base_model)
        model.add(Dropout(.2))

        model.add(Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(.1))
        model.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(.1))
        model.add(Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(.1))
        model.add(Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(.1))
        model.add(Conv2D(500, (3, 3), strides=(1, 1), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, strides=(2, 2), padding='same'))

        # Add new layers
        model.add(Flatten())
        model.add(Dense(FC, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(FC, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(FC, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(2, activation='softmax'))

        model.summary()

        # early stopping to monitor the validation loss and avoid overfitting
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)

        # reducing learning rate on plateau
        rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)

        # model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
        #               loss='categorical_crossentropy',
        #               metrics=['accuracy'])
        # callbacks = [tb(log_dir="logs_vgg16_4")]

        model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        callbacks = [early_stop, rlrop, tb(log_dir="../../logs_vgg16_7")]

        num_train_examples = len(train_images) * 0.8
        BATCH_SIZE = 64

        model.fit(datagen.flow(train_images, train_gender, batch_size=BATCH_SIZE),
                  epochs=50,
                  steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE),
                  shuffle=True, validation_data=(val_images, val_gender),
                  callbacks=callbacks)

        model_json = model.to_json()
        with open(fileJson, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(fileH5)
        print("Saved model to disk")
    else:
        json_file = open(fileJson, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(fileH5)
        model = loaded_model
        print("Loaded model from disk")

        # model.summary()
        #
        # print("")

        model.compile(optimizer=tf.keras.optimizers.Adam(.0001),
                      loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.Recall(),
                               tf.keras.metrics.Precision()])

    # # This is the standard testing on the same dataset but unseen data, this achieves the highest accuracy.
    # # loss: 0.3425 - accuracy: 0.8625
    # print("\nEvaluating the Model\n")
    # model.evaluate(test_images, test_gender)

    results = model.evaluate(test_images, test_gender)

    print(f'Test Acc: {results[1]}\n'
          f'Test Recall: {results[2]}\n'
          f'Test Precision: {results[3]}\n')
    print('')
    print(f'F1-Score: {2 * (results[3] * results[2]) / (results[3] + results[2])}')

    return model

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
    cnn_face_detector = mtcnn.MTCNN()
    faces_cnn = cnn_face_detector.detect_faces(img)
    print(f'This is the number of faces w probability: \n'
          f'{faces_cnn}')

    faces = []
    if (len(faces_cnn) > 0):
        for face in faces_cnn:

            confidence = face['confidence']
            if confidence < .91:
                break

            x, y, width, height = face['box']
            print("face", x, y, width, height)

            face_img = tf.image.crop_to_bounding_box(img,
                                                     y, x,
                                                     height, width)

            face_img = tf.image.resize(face_img, (32, 32), method=tf.image.ResizeMethod.BICUBIC, antialias=True)
            # face_img = tf.image.resize(face_img, (128, 128), method=tf.image.ResizeMethod.BICUBIC, antialias=True)
            face_img = tf.dtypes.cast(face_img, tf.int32)
            faces.append(
                # {'face': face_img.numpy(), 'coordinates': [offset_x, offset_y, target_width, target_height]}
                {'face': face_img.numpy(), 'coordinates': [x, y, width, height]}
            )
    return faces

def show_output(img, faces, predictions):
    for i, data in enumerate(faces):
        coordinates = data['coordinates']
        x1 = coordinates[0]
        y1 = coordinates[1]
        x2 = coordinates[2] + x1
        y2 = coordinates[3] + y1
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        gender = 'Male' if np.argmax(predictions[i]) == 0 else 'Female'
        cv2.putText(img, gender, (x1 - 3, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    plt.imshow(img)
    plt.show()

print(f'Configuring Model...')
start = datetime.now()
model = config_model()
end = datetime.now()
print(f'Model Completed!!!\n'
      f'Time Taken - {end - start}')