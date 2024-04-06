# Ubaid Mahmood
# S1906881
# The stated files will be located in google Drive.

import tensorflow as tf
from keras.callbacks import TensorBoard as tb
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import mtcnn.mtcnn
import sys

# This class is utilised to construct and configure the gender detection model.
class GenderDetection:

    image = None

    # The function configures the model.
    # Returns - model.
    def config_model(self):

        # This will remove the warnings from tensorflow about AVX
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        print(f"Tensor Flow Version: {tf.__version__}")
        print(f"Keras Version: {tf.keras.__version__}")
        print()
        print(f"Python {sys.version}")
        gpu = len(tf.config.list_physical_devices('GPU')) > 0
        print("GPU is", "available" if gpu else "NOT AVAILABLE")

        data = open(os.path.abspath("processedData/wiki_data.pickle"), "rb")
        data = pickle.load(data)
        open(os.path.abspath("processedData/wiki_data.pickle"), "rb").close()

        datagen = ImageDataGenerator(zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                     horizontal_flip=True)

        images = np.array(data['images'])
        genders = np.array(data['gender'])

        print('''The shape of the images array is : {}\n
                The shape is an image is : {}\n
                The shape of the gender array is : {}'''.format(images.shape, images[0].shape, genders.shape))


        # This makes the amount of female and male in the dataset equal.
        f = 0
        m = 0
        index = 0
        mod_genders = []
        mod_images = []
        males_images = []
        males_genders = []
        for z in genders:
            if z == 0.0:
                mod_images.append(images[index])
                mod_genders.append(genders[index])
                f += 1
            elif m <= f:
                # print("")
                mod_images.append(images[index])
                mod_genders.append(genders[index])
                males_images.append(images[index])
                males_genders.append(genders[index])
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
                The shape of the gender array is : {}'''.format(mod_images.shape, mod_images[0].shape,
                                                                mod_genders.shape))

        genderCategorical = []

        for i in mod_genders:
            if i[0] == 1:
                genderCategorical.append([1.0, 0.0])
            else:
                genderCategorical.append([0.0, 1.0])
        genderCategorical = np.array(genderCategorical)

        train_images, val_images, train_gender, val_gender = train_test_split(mod_images, genderCategorical,
                                                                              test_size=.2, shuffle=True,
                                                                              random_state=10)

        train_images, test_images, train_gender, test_gender = train_test_split(train_images, train_gender,
                                                                                test_size=.2, shuffle=True,
                                                                                random_state=10)

        # This file has a model that achieves 92.15% accuracy.
        file = "models/final_model_vgg16.json"
        fileJson = "models/final_model_vgg16.json"
        fileH5 = "models/final_model_vgg16.h5"

        print(f'The model being utilised is: {file}')

        if not os.path.exists(os.path.abspath(file)):

            FC = 2048
            E = 1
            fre = -20

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

            model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            callbacks = [tb(log_dir="logs_new_vgg16_7")]

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

            model.summary()

            print("")

            model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                          loss='binary_crossentropy',
                          metrics=['accuracy', tf.keras.metrics.Recall(),
                               tf.keras.metrics.Precision()])

        # This is the standard testing on the same dataset but unseen data, this achieves the highest accuracy.
        print("\nEvaluating the Model\n")
        results = model.evaluate(test_images, test_gender)

        print(f'Test Acc: {results[1]}\n'
              f'Test Recall: {results[2]}\n'
              f'Test Precision: {results[3]}\n')
        print(f'F1-Score: {2 * (results[3] * results[2]) / (results[3] + results[2])}')

        return model

    # This function extracts faces from the specified image.
    # Return faces - this array contains all the faces identified.
    def extract_faces(self, img):
        cnn_face_detector = mtcnn.MTCNN()
        faces_cnn = cnn_face_detector.detect_faces(img)
        faces = []
        if (len(faces_cnn) > 0):
            for face in faces_cnn:

                x, y, width, height = face['box']
                print("face", x, y, width, height)

                face_img = tf.image.crop_to_bounding_box(img,
                                                         y, x,
                                                         height, width)

                face_img = tf.image.resize(face_img, (32, 32), method=tf.image.ResizeMethod.BICUBIC, antialias=True)
                face_img = tf.dtypes.cast(face_img, tf.int32)
                faces.append(
                    {'face': face_img.numpy(), 'coordinates': [x, y, width, height]}
                )

        return faces

    # This function displays the image with the predicted gender.
    #
    def show_output(self, img, faces, predictions):
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

    # This method states if the model predicted male or female.
    # Return gender - This is the gender predicted by the model.
    def predicted_gender(self, prediction):
        gender = None

        print(f'\n'
              f'Gender prediction: {prediction}')

        if prediction[0] > prediction[1]:
            print(f'Its a Male')
            gender = "Male"
        else:
            print(f'Its a Female')
            gender = "Female"

        return gender

    # After the model is configured, this method is invoked to run the model.
    # Return prediction_gender - This is the predicted gender.
    def run(self, model, ran_img):
        faces = self.extract_faces(ran_img)
        # print(faces)

        prediction_gender = None

        predict_X = []
        for face in faces:
            predict_X.append(face['face'])

        predict_X = np.array(predict_X)

        predictions = []
        if predict_X.shape[0] > 0:
            predictions = model.predict(predict_X)

        print(f'This is the prediction guess of it being female or male: \n'
              f'{predictions}')


        if len(predictions) == 0:
            print(f'list is empty')
        else:

            prediction_gender = self.predicted_gender(predictions[0])

        return prediction_gender

