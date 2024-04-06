# Ubaid Mahmood
# S1906881

# Import Libraries
import numpy as np
from scipy.io import loadmat
import pandas as pd
import datetime as date
from dateutil.relativedelta import relativedelta

# This file converts the mat file into csv,
# this will allow for data visualisation and manipulation.

cols = ['age', 'gender', 'path', 'face_score1', 'face_score2']

wiki_mat = 'wiki/wiki.mat'
wiki_data = loadmat(wiki_mat)

del wiki_mat

wiki = wiki_data['wiki']

wiki_photo_taken = wiki[0][0][1][0]
wiki_full_path = wiki[0][0][2][0]
wiki_gender = wiki[0][0][3][0]
wiki_face_score1 = wiki[0][0][6][0]
wiki_face_score2 = wiki[0][0][7][0]

wiki_path = []

for path in wiki_full_path:
    wiki_path.append('wiki/' + path[0])

wiki_genders = []

for n in range(len(wiki_gender)):
    if wiki_gender[n] == 1:
        wiki_genders.append('male')

    else:
        wiki_genders.append('female')

wiki_dob = []

for file in wiki_path:
    wiki_dob.append(file.split('_')[1])

wiki_age = []

print(wiki_dob[1])
for i in range(len(wiki_dob)):
    try:
        d1 = date.datetime.strptime(wiki_dob[i][0:10], '%Y-%m-%d')
        d2 = date.datetime.strptime(str(wiki_photo_taken[i]), '%Y')
        rdelta = relativedelta(d2, d1)
        diff = rdelta.years

    except Exception as ex:
        diff = -1

    wiki_age.append(diff)

final_wiki = np.vstack((wiki_age, wiki_genders, wiki_path, wiki_face_score1, wiki_face_score2)).T
final_wiki_df = pd.DataFrame(final_wiki)
final_wiki_df.columns = cols

wiki_meta = final_wiki_df
wiki_meta = wiki_meta[wiki_meta['face_score1'] != '-inf']
wiki_meta = wiki_meta[wiki_meta['face_score2'] == 'nan']
wiki_meta = wiki_meta.drop(['face_score1', 'face_score2'], axis=1)
wiki_meta = wiki_meta.sample(frac=1)

wiki_meta.to_csv('wiki_meta.csv', index=False)