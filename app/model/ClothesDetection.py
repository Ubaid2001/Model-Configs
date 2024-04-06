# Ubaid Mahmood
# S1906881
# The stated files will be located in google Drive.

# Import Libraries.
import numpy
import pandas as pd
import os
import tensorflow as tf
import pickle
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import TensorBoard as tb
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import efficientnet.keras as efn
import math

# This class will configure the Clothes Detection model, required for the pipeline.
class ClothesDetection:
    global IMG_SIZE

    IMG_SIZE = 224

    def config_model(self):

        folder = "./fashionImages"
        files = os.listdir(folder)

        pickle_file = "processedData/fashionImagesDC12.pickle"
        # pickle_file = "../../processedData/fashionImagesDC13.pickle"

        # If the required pickle file does not exist then.
        # A pickle file will be created from the specified dataset.
        if not os.path.exists(os.path.abspath(pickle_file)):

            df = pd.read_csv("fashionData/dataConfig12.csv")
            # df = pd.read_csv("../../fashionData/dataConfig13.csv")

            images = []
            clothes = []

            print(f'Processing data...')
            for i, file in enumerate(files):
                file2 = file.replace(".jpg", "")
                print(file2)
                integer = int(file2)
                for j, id in enumerate(df['id']):
                    if id == integer:

                        img = cv2.imread(folder + '/' + file)
                        img = cv2.resize(img, (224, 224))
                        images.append(img)
                        clothes.append(df.iloc[j, 4])

            print(f'Processing Complete!!!')

            meta_data = {
                "images": images,
                "clothes": clothes
            }

            binary_file = open(pickle_file, "ab")
            pickle.dump(meta_data, binary_file)
            binary_file.close()
        else:

            data = open(os.path.abspath(pickle_file), "rb")
            data = pickle.load(data)
            open(os.path.abspath(pickle_file), "rb").close()

            images = np.array(data['images'])
            clothes = np.array(data['clothes'])

            print(f'''The shape of the images array is : {images.shape}\n
                    The shape is an image is : {images[0].shape}\n
                    The shape of the clothes array is : {clothes.shape}''')

            # This is for DC12 and DC13 as the classes match
            class_names = ["Tshirts", "Briefs", "Shirts", "Shorts", "Jeans", "Tops", "Trousers", "Bra", "Track Pants",
                           "Innerwear Vests"]

            num_of_classes = len(class_names)

            datagen_train = ImageDataGenerator(zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                                               shear_range=0.15,
                                               horizontal_flip=True, data_format='channels_last',
                                               preprocessing_function=efn.preprocess_input)

            datagen_val = ImageDataGenerator(data_format='channels_last', preprocessing_function=efn.preprocess_input)

            clothesCategorical = []

            # This is for DC12 and DC13 pickle files
            for i in clothes:
                if i == "Tshirts":
                    clothesCategorical.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                elif i == "Briefs":
                    clothesCategorical.append([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                elif i == "Shirts":
                    clothesCategorical.append([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                elif i == "Shorts":
                    clothesCategorical.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                elif i == "Jeans":
                    clothesCategorical.append([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                elif i == "Tops":
                    clothesCategorical.append([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
                elif i == "Trousers":
                    clothesCategorical.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                elif i == "Bra":
                    clothesCategorical.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
                elif i == "Track Pants":
                    clothesCategorical.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                elif i == "Innerwear Vests":
                    clothesCategorical.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            clothesCategorical = np.array(clothesCategorical)

            batch_size = 64

            train_images, val_images, train_clothes, val_clothes = train_test_split(images, clothesCategorical,
                                                                                    test_size=.2, shuffle=True,
                                                                                    random_state=10)

            train_images, test_images, train_clothes, test_clothes = train_test_split(train_images, train_clothes,
                                                                                      test_size=.2, shuffle=True,
                                                                                      random_state=10)

            num_train_examples = len(train_images) * 0.8

            print(f"Number of training images samples: {train_images.shape}")
            print(f"Number of training clothes samples: {train_clothes.shape}")
            print(f"Number of test images samples: {test_images.shape}")
            print(f"Number of test clothes samples: {test_clothes.shape}")

            # This is the model trained with the data generator of DC12 images ver 2.0
            # Changed learning rate to .001
            model_name = "efficientNet_models/new_model_ecnDC12_dg_LTF_v2.json"
            model_h5 = "efficientNet_models/new_model_ecnDC12_dg_LTF_v2.h5"

            # # This is the model trained with the data generator of DC13 images
            # # Changed learning rate to .0001
            # model_name = "efficientNet_models/new_model_ecnDC13_dg_LTF.json"
            # model_h5 = "efficientNet_models/new_model_ecnDC13_dg_LTF.h5"

            print(f'The model being utilised is: {model_name}')

            # If the model exists then it is loaded into the program.
            # Other-wise it is constructed and trained on the dataset.
            if not os.path.exists(os.path.abspath(model_name)):

                base_model = efn.EfficientNetB0(weights='imagenet', include_top=False,
                                                input_shape=(IMG_SIZE, IMG_SIZE, 3), classes=num_of_classes)

                # base_model.summary()

                base_model.trainable = False

                model = Sequential()
                model.add(base_model)
                model.add(GlobalAveragePooling2D())
                model.add(Dropout(0.5))
                model.add(Dense(num_of_classes, activation='softmax'))

                model.summary()

                optimizer = Adam(learning_rate=0.0001)

                # optimizer = Adam(learning_rate=0.001)

                # early stopping to monitor the validation loss and avoid overfitting
                early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10,
                                           restore_best_weights=True)

                # reducing learning rate on plateau
                rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6,
                                          verbose=1)
                # model compiling
                model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

                print("#####################################################################")
                print("Model Training")

                # LTF stands for layers trainable is FALSE: LTF
                model.fit(datagen_train.flow(train_images, train_clothes, batch_size=batch_size),
                          epochs=100,
                          steps_per_epoch=math.ceil(num_train_examples / batch_size),
                          shuffle=True,
                          validation_data=datagen_val.flow(val_images, val_clothes, batch_size=batch_size),
                          callbacks=[early_stop, rlrop, tb(log_dir="../../ECN-Logs-DC13-dg-LTF")])

                model_json = model.to_json()
                with open(model_name, "w") as json_file:
                    json_file.write(model_json)

                model.save_weights(model_h5)
                print("Saved model to disk")
            else:

                json_file = open(model_name, 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = model_from_json(loaded_model_json)

                # load weights into new model
                loaded_model.load_weights(model_h5)
                model = loaded_model
                print("Loaded model from disk")

                loaded_model.summary()

                # optimizer = Adam(learning_rate=0.0001)

                optimizer = Adam(learning_rate=0.001)

                model.compile(optimizer=optimizer,
                              loss='categorical_crossentropy',
                              metrics=['accuracy', tf.keras.metrics.Recall(),
                                       tf.keras.metrics.Precision()])

                print("\nEvaluating the Model\n")
                results = model.evaluate(datagen_val.flow(test_images, test_clothes))

                print(f'Test Acc: {results[1]}\n'
                      f'Test Recall: {results[2]}\n'
                      f'Test Precision: {results[3]}\n')
                print(f'F1-Score: {2 * (results[3] * results[2]) / (results[3] + results[2])}')

                return model

    #
    def make_prediction(self, image, model):

        nImg = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        nImg = nImg[np.newaxis, :, :, :]

        nImg = efn.preprocess_input(nImg)
        predictions = model.predict(nImg)

        print(predictions[0])

        # This is for DC12 and DC13 images
        clothes_dict = {"Tshirts": 0, "Briefs": 0, "Shirts": 0, "Shorts": 0, "Jeans": 0, "Tops": 0, "Trousers": 0,
                        "Bra": 0, "Track Pants": 0, "Innerwear Vests": 0, "No Clothing Item": 0}

        index = 0
        pred = np.copy(predictions[0])
        preds = predictions[0]

        largest_pb = max(preds)

        # If prediction probability is not above the .35 threshold then set prediction of No Clothing Item.
        if largest_pb < .35:
            print(f'the largest prediction is less than .35')
            for pb in range(len(preds)):
                preds[pb] = 0.0
            preds = numpy.append(preds, 1.0)
        else:
            print(f'the largest prediction is greater than .35')
            for pb in range(len(preds)):
                if preds[pb] == largest_pb:
                    preds[pb] = 1.0
                    # print(f'This is the largest value: {c[pb]}, {largest_pb}')
                else:
                    preds[pb] = 0.0
                    # print(f'This is not the largest value: {c[pb]}')
            preds = numpy.append(preds, 0.0)

        for key in clothes_dict:
            clothes_dict[key] = preds[index]
            index += 1

        print(clothes_dict)

        print(len(preds))

        result = ""
        for key in clothes_dict:
            if clothes_dict[key] == 1.0:
                result = key

        print(result)

        return result, pred
