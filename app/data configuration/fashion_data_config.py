# Ubaid Mahmood
# S1906881

# Import Libraries
import pandas as pd
import os
import glob
import shutil
import matplotlib.pyplot as plt

# This python file configures the dataset from the FPI dataset.

csv_file_path = "../../fashionData/styles.csv"

# These files can be found in google drive.
# data_config_csv = "./fashionData/dataConfig12.csv"
data_config_csv = "./fashionData/dataConfig13.csv"

# Path to fashion images folder which should be downloaded on local computer.
folder_path = ""
dir_path = ""

if not os.path.exists(data_config_csv):

    data = pd.read_csv(csv_file_path, on_bad_lines='skip')
    print(f'{data}')
    print(len(data))

    print(f'{data.head(2)}')
    print(f'{data["articleType"]}')

    data_dropped = data.drop(['gender', 'usage', 'productDisplayName', 'baseColour', 'year', 'season'], axis=1)
    print(f'{data_dropped.head(3)}')

    data_dropped = data_dropped[data_dropped.masterCategory != "Accessories"]
    data_dropped = data_dropped[data_dropped.subCategory != "Sandal"]
    data_dropped = data_dropped[data_dropped.masterCategory != "Personal Care"]
    data_dropped = data_dropped[data_dropped.masterCategory != "Free Items"]
    data_dropped = data_dropped[data_dropped.masterCategory != "Sporting Goods"]
    data_dropped = data_dropped[data_dropped.masterCategory != "Home"]
    data_dropped = data_dropped[data_dropped.masterCategory != "Footwear"]
    data_dropped = data_dropped[data_dropped.subCategory != "Dress"]
    data_dropped = data_dropped[data_dropped.subCategory != "Loungewear and Nightwear"]
    data_dropped = data_dropped[data_dropped.subCategory != "Saree"]
    data_dropped = data_dropped[data_dropped.subCategory != "Apparel Set"]
    data_dropped = data_dropped[data_dropped.subCategory != "Socks"]
    data_dropped = data_dropped[data_dropped.articleType != "Nehru Jackets"]
    data_dropped = data_dropped[data_dropped.articleType != "Shrug"]
    data_dropped = data_dropped[data_dropped.articleType != "Salwar and Dupatta"]
    data_dropped = data_dropped[data_dropped.articleType != "Blazers"]
    data_dropped = data_dropped[data_dropped.articleType != "Shapewear"]
    data_dropped = data_dropped[data_dropped.articleType != "Tights"]
    data_dropped = data_dropped[data_dropped.articleType != "Rompers"]
    data_dropped = data_dropped[data_dropped.articleType != "Swimwear"]
    data_dropped = data_dropped[data_dropped.articleType != "Waistcoat"]
    data_dropped = data_dropped[data_dropped.articleType != "Rain Jacket"]
    data_dropped = data_dropped[data_dropped.articleType != "Tracksuits"]
    data_dropped = data_dropped[data_dropped.articleType != "Churidar"]
    data_dropped = data_dropped[data_dropped.articleType != "Stockings"]
    data_dropped = data_dropped[data_dropped.articleType != "Salwar"]
    data_dropped = data_dropped[data_dropped.articleType != "Jeggings"]
    data_dropped = data_dropped[data_dropped.articleType != "Patiala"]
    data_dropped = data_dropped[data_dropped.articleType != "Camisoles"]
    data_dropped = data_dropped[data_dropped.articleType != "Suspenders"]
    data_dropped = data_dropped[data_dropped.articleType != "Dupatta"]
    data_dropped = data_dropped[data_dropped.articleType != "Capris"]
    data_dropped = data_dropped[data_dropped.articleType != "Kurtis"]
    data_dropped = data_dropped[data_dropped.articleType != "Shapewear"]
    data_dropped = data_dropped[data_dropped.articleType != "Kurtas"]
    data_dropped = data_dropped[data_dropped.articleType != "Lehenga Choli"]
    data_dropped = data_dropped[data_dropped.articleType != "Belts"]
    data_dropped = data_dropped[data_dropped.articleType != "Dresses"]
    data_dropped = data_dropped[data_dropped.articleType != "Rain Trousers"]
    data_dropped = data_dropped[data_dropped.articleType != "Suits"]
    data_dropped = data_dropped[data_dropped.articleType != "Jackets"]

    data_dropped = data_dropped.replace('Sweatshirts', 'Tshirts')
    data_dropped = data_dropped.replace('Sweaters', 'Tshirts')
    data_dropped = data_dropped.replace('Skirts', 'Shorts')
    data_dropped = data_dropped.replace('Leggings', 'Track Pants')

    data_dropped = data_dropped[data_dropped.articleType != "Tunics"]
    data_dropped = data_dropped.replace('Boxers', 'Trunk')
    data_dropped = data_dropped.replace('Trunk', 'Briefs')

    data_dropped.drop(data_dropped[(data_dropped.subCategory == "Topwear")].index[0:7999], inplace=True)
    data_dropped.drop(data_dropped[(data_dropped.subCategory == "Topwear")].index[0:2300], inplace=True)

    print(f'The amount of files in the configured data {len(data_dropped)}')
    data_dropped.to_csv(data_config_csv)

else:
    data = pd.read_csv(data_config_csv, on_bad_lines='skip')

    ids = data['id']
    topwear = data[data['subCategory'].values == "Topwear"]
    print(f'This is the length of topwear {len(topwear)}')
    bottomwear = data[data['subCategory'].values == "Bottomwear"]
    print(f'This is the length of bottomwear {len(bottomwear)}')
    inner = data[data['subCategory'].values == "Innerwear"]
    print(f'This is the length of Inner wear {len(inner)}')

    Tshirts = data[data['articleType'].values == "Tshirts"]
    print(f'This is the length of Tshirts {len(Tshirts)}')
    briefs = data[data['articleType'].values == "Briefs"]
    print(f'This is the length of Briefs {len(briefs)}')
    shorts = data[data['articleType'].values == "Shorts"]
    print(f'This is the length of Shorts {len(shorts)}')

    data.subCategory.value_counts().nlargest(40).plot(kind='bar', figsize=(20, 15))
    plt.title("Number of sub Categories")
    plt.ylabel("Number of Items by subCategories")
    plt.xlabel("SubCategories")
    plt.show()

    data.articleType.value_counts().nlargest(40).plot(kind='bar', figsize=(20, 15))
    plt.title("Number of articleType")
    plt.ylabel("Number of Items by articleType")
    plt.xlabel("articleType")
    plt.show()


    if not os.path.exists("../../fashionImages"):
        image_paths = []
        dirPath = None
        print(f'Processing Images...')
        for directory_path in glob.glob(folder_path):

            for i, img_path in enumerate(glob.glob(os.path.join(directory_path, "*.*"))):

                dirPath = directory_path
                image_paths.append(img_path)

        print(f'Processing Complete!!!')

        ids = ids.values

        os.mkdir("../../fashionImages")

        for image_path in image_paths:
            image_id = image_path.replace(dirPath + "\\", "")
            image_id = int(image_id.replace(".jpg", ""))

            for x in ids:
                if x == image_id:

                    shutil.copy2(image_path, "../../fashionImages/")

        print(f'image paths length: {len(image_paths)}')
        print(f'image ids length: {len(ids)}')

    data.subCategory.value_count().nlargest(40).plot(kind='bar', figsize=(10,5))
    plt.title("Number of sub Categories")
    plt.ylabel("Number of Items by subCategories")
    plt.xlabel("SubCategories")
    plt.show()

