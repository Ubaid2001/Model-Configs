# Ubaid Mahmood
# S1906881
# The stated files will be located in google Drive.

# Import Libraries
import tensorflow as tf
import os
import pickle
import scipy.io
import bz2
import datetime
import math
import cv2
import sys
import mtcnn.mtcnn

# This will remove the warnings from tensorflow about AVX
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print()
print(f"Python {sys.version}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

# Retrieves the mmod face detector which has now been replaced with MTCNN
if not os.path.exists(os.path.abspath('') + 'mmod_human_face_detector.dat.bz2'):
        annotation_zip = tf.keras.utils.get_file('mmod_human_face_detector.dat.bz2',
                                                 cache_subdir=os.path.abspath(''),
                                                 origin="http://dlib.net/files/mmod_human_face_detector.dat.bz2")


# Using pythons bz2 package to read the bz2 file in binary format and write it into a .dat file
with bz2.open("mmod_human_face_detector.dat.bz2", "rb") as f:
  content = f.read()

  with open("../../mmod_human_face_detector.dat", "wb") as weights_file:
    weights_file.write(content)

os.remove(annotation_zip)


dataset_url = 'https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar'
annotation_folder = "wiki_crop"

# Retrieves the wiki_crop dataset from IMDB/WIKI website.
if not os.path.exists('./' + annotation_folder):
    annotation_zip = tf.keras.utils.get_file('wiki_crop.tar',
                                             cache_subdir=os.path.abspath(''),
                                             origin=dataset_url,
                                             extract=True)
    os.remove(annotation_zip)

data_key = 'wiki'
mat_file = 'wiki.mat'

mat = scipy.io.loadmat(annotation_folder + '/' + mat_file)
data = mat[data_key]
route = data[0][0][2][0]
name = []
age = []
gender = []
images = []

total = 0
project_path = "/"

total = 0
cnn_face_detector = mtcnn.MTCNN()
today = datetime.date.today()

# Loops through the dataset and pre-processes the images.
for i in range(0, len(route)):
    if i % 100 == 0:
        print(i)
    try:
        if ((math.isnan(data[0][0][6][0][i]) == False and data[0][0][6][0][i] > 0) and math.isnan(
                data[0][0][3][0][i]) == False):
            img = cv2.imread(annotation_folder + "/" + route[i][0])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            faces_cnn = cnn_face_detector(img, 1)

            if len(faces_cnn) == 1:
                total += 1

                x, y, width, height = faces_cnn['box']
                print("face", x, y, width, height)

                face_img = tf.image.crop_to_bounding_box(img,
                                                         y, x,
                                                         height, width)

                face_img = tf.image.resize(face_img, (32, 32), method=tf.image.ResizeMethod.BICUBIC, antialias=True)
                face_img = tf.dtypes.cast(face_img, tf.int32)

                images.append(face_img.numpy())
                temp = datetime.date.fromordinal(int(data[0][0][0][0][i])) - datetime.timedelta(days=366)
                age.append([today.year - temp.year])
                name.append(data[0][0][4][0][i])
                gender.append([data[0][0][3][0][i]])

    except Exception as err:
        print(err)
        print("error in i value ", i)
        if (len(name) == total):
            name.pop()
        if (len(gender) == total):
            gender.pop()
        if (len(age) == total):
            age.pop()
        if (len(images) == total):
            images.pop()
        total -= 1

    except KeyboardInterrupt:
        break

print(total, " elements were processed and stored")

meta_data = {
    "images": images,
    "gender": gender
}

# The data is converted into a pickle file.
if not os.path.exists("../../processedData"):
    os.mkdir(project_path + "/processedData")
    binary_file = open(project_path + "/processedData/wiki_data.pickle", "ab")
    pickle.dump(meta_data, binary_file)
    binary_file.close()