import numpy as np
from scipy.io import loadmat
import pandas as pd
import datetime as date
from dateutil.relativedelta import relativedelta
import cv2

# cols = ['age', 'gender', 'path', 'face_score1', 'face_score2']
cols = ['gender', 'path', 'face_score1', 'face_score2']

imdb_mat = 'imdb/imdb.mat'
wiki_mat = 'wiki/wiki.mat'

imdb_data = loadmat(imdb_mat)
wiki_data = loadmat(wiki_mat)

del imdb_mat, wiki_mat

# print(imdb_data)


# del imdb_mat
# del wiki_mat


imdb = imdb_data['imdb']
wiki = wiki_data['wiki']

imdb_photo_taken = imdb[0][0][1][0]
imdb_full_path = imdb[0][0][2][0]
imdb_gender = imdb[0][0][3][0]
imdb_face_score1 = imdb[0][0][6][0]
imdb_face_score2 = imdb[0][0][7][0]

# print(f"this is the imdb photo taken: {len(imdb_photo_taken)}\n"
#       f"this the imdb full path: {len(imdb_full_path)}\n"
#       f"this is the imdb gender: {len(imdb_gender)}\n"
#       f"this is the imdb face score1: {len(imdb_face_score1)}\n"
#       f"this is the imdb face score2: {len(imdb_face_score2)}")

# new_face_score1 = []
# s1_count = 0
# for s1 in imdb_face_score1:
#     new_face_score1.append(s1)
#     s1_count += 1
#     if s1_count == 10000:
#         break
#
# print(f'this is the new face score1: {len(new_face_score1)}')
#
# new_face_score2 = []
# s2_count = 0
# for s2 in imdb_face_score2:
#     new_face_score2.append(s2)
#     s2_count += 1
#     if s2_count == 10000:
#         break
#
# print(f'this is the new face score2: {len(new_face_score2)}')



wiki_photo_taken = wiki[0][0][1][0]
wiki_full_path = wiki[0][0][2][0]
wiki_gender = wiki[0][0][3][0]
wiki_face_score1 = wiki[0][0][6][0]
wiki_face_score2 = wiki[0][0][7][0]

imdb_path = []
wiki_path = []

for path in imdb_full_path:
    imdb_path.append('imdb/' + path[0])

# p = 0
# for path in imdb_full_path:
#     imdb_path.append('imdb/' + path[0])
#     p += 1
#     if p == 10000:
#         break
# print(len(imdb_path))

for path in wiki_full_path:
    wiki_path.append('wiki/' + path[0])

imdb_genders = []
wiki_genders = []

for n in range(len(imdb_gender)):
    if imdb_gender[n] == 1:
        imdb_genders.append('male')
    else:
        imdb_genders.append('female')

# for n in range(0, 10000):
#     if imdb_gender[n] == 1:
#         imdb_genders.append('male')
#     else:
#         imdb_genders.append('female')
# print(len(imdb_genders))

for n in range(len(wiki_gender)):
    if wiki_gender[n] == 1:
        wiki_genders.append('male')
    else:
        wiki_genders.append('female')

imdb_dob = []
wiki_dob = []

# for file in imdb_path:
#     print(file)
#     temp = file.split('_')[3]
#     temp = temp.split('-')
#     print(temp[0])
#     if len(temp[1]) == 1:
#         temp[1] = '0' + temp[1]
#     if len(temp[2]) == 1:
#         temp[2] = '0' + temp[2]
#
#     if temp[1] == '00':
#         temp[1] = '01'
#     if temp[2] == '00':
#         temp[2] = '01'
#
#     imdb_dob.append('-'.join(temp))

# for file in wiki_path:
#     wiki_dob.append(file.split('_')[2])

imdb_age = []
wiki_age = []

# for i in range(len(imdb_dob)):
#     try:
#         d1 = date.datetime.strptime(imdb_dob[i][0:10], '%Y-%m-%d')
#         d2 = date.datetime.strptime(str(imdb_photo_taken[i]), '%Y')
#         rdelta = relativedelta(d2, d1)
#         diff = rdelta.years
#     except Exception as ex:
#         print(ex)
#         diff = -1
#     imdb_age.append(diff)

# for i in range(len(wiki_dob)):
#     try:
#         d1 = date.datetime.strptime(wiki_dob[i][0:10], '%Y-%m-%d')
#         d2 = date.datetime.strptime(str(wiki_photo_taken[i]), '%Y')
#         rdelta = relativedelta(d2, d1)
#         diff = rdelta.years
#     except Exception as ex:
#         print(ex)
#         diff = -1
#     wiki_age.append(diff)

# final_imdb = np.vstack((imdb_age, imdb_genders, imdb_path, imdb_face_score1, imdb_face_score2)).T


# new_imdb_path = []
# d = 0
# for i in imdb_path:
#     new_imdb_path.append(cv2.imread(i))
#     if d % 100 == 0:
#         print(d)
#     d += 1
#
# new_imdb_path = np.array(new_imdb_path, dtype=object)
# print(len(new_imdb_path))
#
# final_imdb = np.vstack((imdb_genders, new_imdb_path, new_face_score1, new_face_score2)).T
# print(final_imdb)

final_imdb = np.vstack((imdb_genders, imdb_path, imdb_face_score1, imdb_face_score2)).T

# final_wiki = np.vstack((wiki_age, wiki_genders, wiki_path, wiki_face_score1, wiki_face_score2)).T
final_wiki = np.vstack((wiki_genders, wiki_path, wiki_face_score1, wiki_face_score2)).T

final_imdb_df = pd.DataFrame(final_imdb)
final_wiki_df = pd.DataFrame(final_wiki)

final_imdb_df.columns = cols
final_wiki_df.columns = cols

# meta = pd.concat((final_imdb_df, final_wiki_df))
imdb_meta = final_imdb_df
wiki_meta = final_wiki_df

# print(imdb_meta['face_score1'])

imdb_meta = imdb_meta[imdb_meta['face_score1'] != '-inf']
imdb_meta = imdb_meta[imdb_meta['face_score2'] == 'nan']

wiki_meta = wiki_meta[wiki_meta['face_score1'] != '-inf']
wiki_meta = wiki_meta[wiki_meta['face_score2'] == 'nan']

imdb_meta = imdb_meta.drop(['face_score1', 'face_score2'], axis=1)
wiki_meta = wiki_meta.drop(['face_score1', 'face_score2'], axis=1)

imdb_meta = imdb_meta.sample(frac=1)
wiki_meta = wiki_meta.sample(frac=1)

imdb_meta.to_csv('imdb_meta.csv', index=False)
wiki_meta.to_csv('wiki_meta.csv', index=False)