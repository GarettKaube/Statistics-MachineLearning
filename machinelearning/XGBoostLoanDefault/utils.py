import urllib
import urllib.request
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import os
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

def create_dirs():
    try: 
        os.mkdir('./data') 
    except OSError as error:
        print('data folder already exists')  



def download_and_unzip(url, extract_to='.'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)



def create_paths(downloads):
    paths = []
    for dwl in downloads:
        path = os.path.join('./data', f"{dwl.lower()}.csv")
        paths.append(path)
    return paths



def fetch_external_data(paths, roots):
    for r, p in zip(roots, paths):
        urllib.request.urlretrieve(r, p)



def one_hot(data, categorical, labels):
    j = 0
    for cat in categorical:
        
        OH = OneHotEncoder()
        # feature to be encoded
        transform = data[[cat]]
        indecies = transform.index
        
        cats = list(data[cat].unique())
        cats_list = []
        # get categories
        for i in cats:
            cat_ = [i]
            cats_list.append(cat_)
        OH.fit(cats_list)

        # transform categorical data
        transformed = OH.transform(transform).toarray()
        transformed = pd.DataFrame(transformed, columns=labels[j], index=indecies)
        # drop old categorical column
        drop_ = data.drop(cat, axis=1)
        
        # join data
        data = pd.concat([drop_, transformed], axis = 1)
        
        j+=1
        
    return data