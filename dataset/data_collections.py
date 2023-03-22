from time import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import kaggle
import zipfile
from zipfile import ZipFile

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform

def main():
    url_kaggle = "https://www.kaggle.com/datasets/sorokin/faceforensics"
    kaggle.api.authenticate()
  #  url = "https://github.com/ondyari/FaceForensics"
   # df = pd.read_csv(url, index_col=0, parse_dates=[0])
    print("____DF____")
 #   kaggle.api.dataset_download_files('sorokin/faceforensics', path='dataset/data', unzip=False)
    print("download")
    #kaggle datasets download -d sorokin/faceforensics
    #print(df)
    archive = 'dataset/data/faceforensics.zip'
   # zip_file = ZipFile(archive)
    with zipfile.ZipFile(archive, 'r') as zip_file:
        zip_file.extractall('dataset/data/files')

    print("end")

'''   
    lfw_people = fetch_lfw_people(min_faces_per_person=60, resize=0.4)

    # introspect the images arrays to find the shapes (for plotting)
    n_samples, h, w = lfw_people.images.shape

    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    X = lfw_people.data
    n_features = X.shape[1]

    # the label to predict is the id of the person
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)

    df = pd.DataFrame([list(x)+[y[i]] for i, x in enumerate(X)])
    rows_in_file = int(n_samples/n_classes)
    for i in range(n_samples):
        df.iloc[i*rows_in_file:(i+1)*rows_in_file].to_csv(f"./dataset/data/data_{i}.csv", index=False)
'''
if __name__ == "__main__":
    main()
