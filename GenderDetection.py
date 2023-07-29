import random
import tensorflow as tf
from IPython.core.display import Javascript
from IPython.display import display
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


class GenderDetection:

    # global model
    image = None

    # To messy make another function called configure and pass the image parameter to it.
    def __init__(self, image):
        self.image = image
        self.config_model(image)

    def config_model(self, image):
        data = open("processedData/wiki_data.pickle", "rb")
        data = pickle.load(data)
        open("processedData/wiki_data.pickle", "rb").close()

        datagen = ImageDataGenerator(zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                     horizontal_flip=True)

        images = np.array(data['images'])
        genders = np.array(data['gender'])
        print('''The shape of the images array is : {}\n
                The shape is an image is : {}\n
                The shape of the gender array is : {}'''.format(images.shape, images[0].shape, genders.shape))

        # For imdb_new_data.pickle
        # test_data = open("./processedData/imdb_new_data.pickle", "rb")
        # test_data = pickle.load(test_data)
        # open("./processedData/imdb_new_data.pickle", "rb").close()
        #
        # test_data_images = np.array(test_data['images'])
        # test_data_gender = np.array(test_data['gender'])
        # print('''The shape of the images array is : {}\n
        # The shape is an image is : {}'''.format(test_data_images.shape, test_data_images[0].shape))

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
        # print(len(mod_images))

        # For imdb_new_data.pickle
        # genderImdbCategorical = []
        #
        # for i in test_data_gender:
        #     if i[0] == 1:
        #         genderImdbCategorical.append([1.0, 0.0])
        #     else:
        #         genderImdbCategorical.append([0.0, 1.0])
        # genderImdbCategorical = np.array(genderImdbCategorical)

        # This is without the data augmentation tha occurs in the model.fit()
        # train_images, test_images, train_gender, test_gender = train_test_split(mod_images, genderCategorical,
        #                                                                         test_size=.2, shuffle=True, random_state=10)

        # # These 2 train_test_split() is for the data augmentation as the fit() requires the val data.
        # This model model_vgg16 achieves metric of loss: 0.3425 - accuracy: 0.8625
        # train_images, val_images, train_gender, val_gender = train_test_split(mod_images, genderCategorical,
        #                                                                         test_size=.2, shuffle=True, random_state=10)
        #
        # train_images, test_images, train_gender, test_gender = train_test_split(train_images, train_gender,
        #                                                                         test_size=.2, shuffle=True, random_state=10)

        # # This training set with new_model_vgg16 achieves a loss: 0.2986 - accuracy: 0.8830
        # train_images, val_images, train_gender, val_gender = train_test_split(mod_images, genderCategorical,
        #                                                                         test_size=.2, shuffle=True, random_state=10)
        #
        # train_images, test_images, train_gender, test_gender = train_test_split(train_images, train_gender,
        #                                                                         test_size=.2, shuffle=True, random_state=10)

        # This training set with new_model_vgg16(2) achieves a loss: 0.2815 - accuracy: 0.8887
        train_images, val_images, train_gender, val_gender = train_test_split(mod_images, genderCategorical,
                                                                              test_size=.2, shuffle=True,
                                                                              random_state=10)

        train_images, test_images, train_gender, test_gender = train_test_split(train_images, train_gender,
                                                                                test_size=.2, shuffle=True,
                                                                                random_state=10)

        # # This training set with model_vgg16(tv3) achieves a loss: 0.4044 - accuracy: 0.8516
        # train_images, val_images, train_gender, val_gender = train_test_split(mod_images, genderCategorical,
        #                                                                         test_size=.15, shuffle=True, random_state=10)
        #
        # train_images, test_images, train_gender, test_gender = train_test_split(train_images, train_gender,
        #                                                                         test_size=.2, shuffle=True, random_state=10)

        # For imdb_new_data.pickle
        # _, test_imdb_images, _, test_imdb_gender = train_test_split(test_data_images, genderImdbCategorical,
        #                                                                         test_size=.9, shuffle=True, random_state=10)

        # This is for another dataset the retrieves 1000 images and gender labels with is used for testing the model.
        # This is an alternative method.
        #
        # df = pd.read_csv("imdb_meta.csv")
        # # print(df['path'])
        # images_array = df['path']
        # # print(images_array)
        # # images_array = np.array(images_array)
        # # print(images_array)
        # image = cv2.imread(images_array[0])
        # # print(image)
        #
        # t = 0
        # images_repo = []
        # for i in images_array:
        #     a = cv2.imread(i)
        #     a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        #     a = tf.image.resize(a, (32, 32), method=tf.image.ResizeMethod.BICUBIC, antialias=True)
        #     a = tf.dtypes.cast(a, tf.int32)
        #     images_repo.append(a)
        #     t += 1
        #     if t % 100 == 0:
        #         print(t)
        #     if t == 1000:
        #         break
        #
        # # print(images_repo)
        #
        # gender_array = df['gender']
        # # print(gender_array)
        #
        # gendersCate = []
        # gt = 0
        # for i in gender_array:
        #     if i == "male":
        #         gendersCate.append([1.0, 0.0])
        #     else:
        #         gendersCate.append([0.0, 1.0])
        #     gt += 1
        #     if gt == 1000:
        #         break
        # gendersCate = np.array(gendersCate)
        #
        # images_repo = np.array(images_repo)
        # print(images_repo.shape)
        # # im = images_array[0]
        # # print(mod_images[0])
        #
        # _, test_imdb_images, _, test_imdb_gender = train_test_split(images_repo, gendersCate,
        #                                                              test_size=.9, shuffle=True, random_state=10)
        #
        # print(len(test_imdb_images))
        # print(len(test_imdb_gender))

        if not os.path.exists("models/new_model_vgg16(2).json"):

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

            model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            callbacks = [tb(log_dir="logs")]

            # lrd = ReduceLROnPlateau(monitor = 'val_loss',
            #                         patience = patience,
            #                         verbose = verbose ,
            #                         factor = factor,
            #                         min_lr = min_lr)
            #
            # mcp = ModelCheckpoint('model.h5')
            #
            # es = EarlyStopping(verbose=verbose, patience=patience)

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

            # model.fit(datagen.flow(train_images, train_gender, batch_size=BATCH_SIZE),
            #           epochs=15,
            #           steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE),
            #           shuffle=True, validation_data=(val_images, val_gender),
            #           callbacks=callbacks)
            # loss: 0.3460 - accuracy: 0.8419

            model.fit(datagen.flow(train_images, train_gender, batch_size=BATCH_SIZE),
                      epochs=15,
                      steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE),
                      shuffle=True, validation_data=(val_images, val_gender),
                      callbacks=callbacks)

            # print("\nEvaluating the Model\n")
            # model.evaluate(test_images, test_gender, callbacks=callbacks)

            model_json = model.to_json()
            with open("models/new_model_vgg16(2).json", "w") as json_file:
                json_file.write(model_json)

            # serialize weights to HDF5
            model.save_weights("models/new_model_vgg16(2).h5")
            print("Saved model to disk")
        else:
            json_file = open('models/new_model_vgg16(2).json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("models/new_model_vgg16(2).h5")
            model = loaded_model
            print("Loaded model from disk")

            model.compile(optimizer=tf.keras.optimizers.Adam(),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            callbacks = [tb(log_dir="logs")]

        # This is the standard testing on the same dataset but unseen data, this achieves the highest accuracy.
        # loss: 0.3425 - accuracy: 0.8625
        print("\nEvaluating the Model\n")
        model.evaluate(test_images, test_gender, callbacks=callbacks)

        self.run(model, image)

        # This is the evaluation for either of the two testing datasets
        # print("\nEvaluating the Model\n")
        # model.evaluate(test_imdb_images, test_imdb_gender, callbacks=callbacks)

    def take_photo(self):
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

    def extract_faces(self, img):
        cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
        faces_cnn = cnn_face_detector(img, 1)
        faces = []
        if (len(faces_cnn) > 0):
            for face in faces_cnn:
                offset_x, offset_y = max(face.rect.left(), 0), max(face.rect.top(), 0)
                target_width, target_height = face.rect.right() - offset_x, face.rect.bottom() - offset_y

                target_width = min(target_width, img.shape[1] - offset_x)
                target_height = min(target_height, img.shape[0] - offset_y)
                print("face", offset_x, offset_y, target_width, target_height)

                face_img = tf.image.crop_to_bounding_box(img,
                                                         offset_y, offset_x,
                                                         target_height, target_width)

                face_img = tf.image.resize(face_img, (32, 32), method=tf.image.ResizeMethod.BICUBIC, antialias=True)
                face_img = tf.dtypes.cast(face_img, tf.int32)
                faces.append(
                    {'face': face_img.numpy(), 'coordinates': [offset_x, offset_y, target_width, target_height]})
        return faces

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

    # imgW = cv2.imread("women_pics/w10.jpeg")
    # # plt.imshow(imgW)
    # # plt.show()
    # imgW = cv2.cvtColor(imgW, cv2.COLOR_BGR2RGB)
    # # plt.imshow(imgW)
    # # plt.show()
    # # print(imgW.shape)
    # # imgW = cv2.resize(imgW, (32, 32))
    # # print(imgW.shape)
    # faces = extract_faces(imgW)
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
    # print(predictions)
    # show_output(imgW, faces, predictions)

    def run(self, model, ran_img):
        # il = os.listdir("imdb-images/04")
        # il = np.array(il)
        # # print(il[0])
        #
        # ran_img = cv2.imread("./imdb-images/04/" + il[random.choice(range(0, len(il)))])
        # ran_img = cv2.cvtColor(ran_img, cv2.COLOR_BGR2RGB)
        # plt.imshow(ran_img)
        # plt.show()

        # ran_img = cv2.imread("man_pics/man2.jpg")
        # ran_img = cv2.cvtColor(ran_img, cv2.COLOR_BGR2RGB)
        # print(ran_img.shape)

        faces = self.extract_faces(ran_img)
        # print(faces)

        predict_X = []
        for face in faces:
            predict_X.append(face['face'])

        predict_X = np.array(predict_X)

        predictions = []
        if predict_X.shape[0] > 0:
            predictions = model.predict(predict_X)

        print(f'This is the prediction guess of it being female or male: \n'
              f'{predictions}')
        self.show_output(ran_img, faces, predictions)

    # img = take_photo()
    # # print(img)
    # img = cv2.imread('opencv_frame_0.png')
    # image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # print(image_rgb.shape)
    # # image_rgb = cv2.resize(image_rgb, (255, 255))
    # # print(image_rgb.shape)
    #
    # faces = extract_faces(image_rgb)
    # # print(faces)
    # predict_X = []
    # for face in faces:
    #     predict_X.append(face['face'])
    #
    # predict_X = np.array(predict_X)
    # predictions = []
    # if predict_X.shape[0] > 0:
    #     predictions = model.predict(predict_X)
    #
    # print(predictions)
    # show_output(image_rgb, faces, predictions)
