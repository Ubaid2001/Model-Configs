import pickle
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# file = open('processedData/wiki_data.pickle', 'rb')
#
# data = pickle.load(file)
#
# file.close()
#
# print('Showing the pickled data: ')
#
# for d in data['age']:
#     print(f'The data: {d}')


# print(f'The data: {data["images"]}')

# images = np.array(data['images'])
# gender = np.array(data['gender'])
# print(f'''The shape of the images array is : {images.shape}\n
# The shape is an image is : {images[0].shape}\n
# The shape of the gender array is : {gender.shape}''')


df = pd.read_csv("imdb_meta.csv")
# print(df['path'])
images_array = df['path']
# print(images_array)
# images_array = np.array(images_array)
# print(images_array)
image = cv2.imread(images_array[0])
# print(image)

t = 0
images_repo = []
for i in images_array:
    a = cv2.imread(i)
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    images_repo.append(a)
    t += 1
    if t % 100 == 0:
        print(t)
    if t == 1000:
        break

print(images_repo)

gender_array = df['gender']
print(gender_array)

genderCategorical = []

for i in gender_array:
    if i == "male":
        genderCategorical.append([1.0, 0.0])
    else:
        genderCategorical.append([0.0, 1.0])
genderCategorical = np.array(genderCategorical)

# print(genderCategorical)
img = images_repo[0]
print(img)
plt.title(genderCategorical[0])
plt.imshow(img)
plt.show()