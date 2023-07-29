# This is a sample Python script.
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import cv2

from GenderDetection import GenderDetection
from Movenet import Movenet


# import sys
# import random
# from tensorflow.python.client import device_lib
# from numba import jit, cuda

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    frame = cv2.imread("man_pics/man1.jpg")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    model = GenderDetection(frame)
    pose_model = Movenet()
    pose_model.get_image(frame)

# def plot_gender():
#     data = pd.read_csv('wiki_meta.csv')
#     print(data.head())
#     print(data.columns)
#     print(data.shape)
#
#     gender = []
#     for g in data['gender'].values:
#         if g == 'male':
#             gender.append(1)
#         else:
#             gender.append(0)
#
#     plt.hist(gender, range(3))
#     plt.title(
#         'There are total ' + str(len(gender) - sum(gender)) + ' female images and ' + str(sum(gender)) + ' male images')
#     plt.show()

    # path = data['path'].values
    #
    # num = random.randint(1, len(path))
    #
    # print(path[num])
    #
    # img = cv2.imread(data['path'].values[num], cv2.IMREAD_ANYCOLOR)
    #
    # while True:
    #     gend = str(data['gender'].values[num])
    #     cv2.imshow(gend, img)
    #     cv2.waitKey(0)
    #     sys.exit()  # to exit from all the processes
    #
    # cv2.destroyAllWindows()  # destroy all windows



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
