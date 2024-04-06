import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

df = pd.read_csv("../../csvs/imdb_meta.csv")
images_array = df['path']
image = cv2.imread(images_array[0])

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

img = images_repo[0]
print(img)
plt.title(genderCategorical[0])
plt.imshow(img)
plt.show()