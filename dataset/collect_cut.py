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

if __name__ == "__main__":
    main()
