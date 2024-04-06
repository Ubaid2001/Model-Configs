# Ubaid Mahmood
# S1906881

# Import Libraries
import tensorflow as tf
import cv2
import sys
import os

from app.model.GenderDetection import GenderDetection
from app.model.Movenet import Movenet
from app.model.ClothesDetection import ClothesDetection


# This function determines if the image should be blocked, according to specific standards.
def block_image(result1, result2, gender):
    isBlock = False

    if gender == "Male":
        print(f'This is a Male')
        if result1 == "No Clothing Item":
            print(f'This is image is BLOCKED!!!!!!!!!')
            print(f'Blocked beacause male is wearing: {result1}')
            isBlock = True

        if result2 == "Briefs" or result2 == "No Clothing Item":
            print(f'This is image is BLOCKED!!!!!!!!!')
            print(f'Blocked beacause male is wearing: {result2}')
            isBlock = True
    elif gender == "Female":
        print(f'This is a Female')
        if result1 == "Bra" or result1 == "Innerwear Vests" or result1 == "No Clothing Item":
            print(f'This is image is BLOCKED!!!!!!!!!')
            print(f'Blocked beacause female is wearing: {result1}')
            isBlock = True

        if result2 == "Shorts" or result2 == "Briefs" or result2 == "No Clothing Item":
            print(f'This is image is BLOCKED!!!!!!!!!')
            print(f'Blocked beacause female is wearing: {result2}')
            isBlock = True

    print(f'isBlock: {isBlock}\n'
          f'\n')

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def run():

    gender_detection = GenderDetection()
    Gmodel = gender_detection.config_model()
    pose_model = Movenet()
    clothes_detection = ClothesDetection()
    Cmodel = clothes_detection.config_model()

    # The loop goes through a batch of images saved locally.
    # The images are fed into the pipeline.
    # To identify whether to block or display
    for i in range(1, 31):

        frame = cv2.imread(f"C:\\Users\\ubaid\\Documents\\women_pics\\Fully Clothed\\{i}.jpg")
        # frame = cv2.imread(f"women_pics\\Haram\\{i}.jpg")
        #
        # frame = cv2.imread(f"men_pics - Copy\\Fully Clothed\\{i}.jpg")
        # frame = cv2.imread(f"men_pics - Copy\\Haram\\{i}.jpg")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame2 = frame.copy()

        # gender_detection = GenderDetection()
        # Gmodel = gender_detection.config_model()

        print(f'{i}.jpg')

        blockPrint()

        predicted_gender = gender_detection.run(Gmodel, frame)

        if predicted_gender is not None:

            image1, image2 = pose_model.get_image(frame2)
            enablePrint()

            if image1 is not None and image2 is not None:

                blockPrint()
                res1, preds1 = clothes_detection.make_prediction(image1, Cmodel)
                res2, preds2 = clothes_detection.make_prediction(image2, Cmodel)
                enablePrint()

                print(f'res1 - {res1} \n'
                      f'res2 - {res2}')

                block_image(res1, res2, predicted_gender)

            else:
                print(f'\n'
                      f'Since, image1 and image2 is None\n'
                      f'Clothes Detection cannot advance\n'
                      f'Find another full body image\n'
                      f'\n')
        else:
            print(f'prediction is None')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # This will remove the warnings from tensorflow about AVX
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    print()
    print(f"Python {sys.version}")
    gpu = len(tf.config.list_physical_devices('GPU')) > 0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")

    run()

