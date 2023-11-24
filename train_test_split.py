import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import csv


df = pd.read_csv('./data/features_30_sec.csv')

data_path = './data/genres_original/'

train_array = []
test_array = []
val_array = []
for folder in os.listdir(data_path):
    # print(folder)
    files=[]
    for file in os.listdir(data_path+folder):
        # print(file)
        files.append((file, file[:-10]))
    train, test = train_test_split(files, test_size=0.2)
    test, val = train_test_split(test, test_size=0.5)
    # print(train)
    train_array.extend(train)
    test_array.extend(test)
    val_array.extend(val)

with open('train.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['file', 'label'])
    for row in train_array: 
        writer.writerow(row) 

with open('test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['file', 'label'])
    for row in test_array: 
        writer.writerow(row) 

with open('val.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['file', 'label'])
    for row in val_array: 
        writer.writerow(row) 