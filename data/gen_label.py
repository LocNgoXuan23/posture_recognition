import sys
sys.path.append('../src')
from utils import read_json, write_json
import os

# Gen label train
train_data = []
root_train = 'FOLD_1/train'
for i in range(len(os.listdir(root_train))): 
    # root_train_class = os.path.join(root_train, os.listdir(root_train)[i])
    root_train_class = f'{root_train}/{os.listdir(root_train)[i]}'
    # print(root_train_class)
    for j in range(len(os.listdir(root_train_class))):
        # root_class = os.path.join(root_train_class, os.listdir(root_train_class)[j])
        # root_class = os.path.join('..', 'data', root_class)
        root_class = f'../data/{root_train_class}/{os.listdir(root_train_class)[j]}'
        print(root_class)
        train_data.append(root_class)

print(len(train_data))
write_json("train.json", train_data)

val_data = []
# root_val = os.path.join('FOLD_1', 'val')
root_val = 'FOLD_1/val'

for i in range(len(os.listdir(root_val))): 
    # root_val_class = os.path.join(root_val, os.listdir(root_val)[i])
    root_val_class = f'{root_val}/{os.listdir(root_val)[i]}'
    # print(root_val_class)
    for j in range(len(os.listdir(root_val_class))):
        # root_class = os.path.join(root_val_class, os.listdir(root_val_class)[j])
        # root_class = os.path.join('..', 'data', root_class)
        root_class = f'../data/{root_val_class}/{os.listdir(root_val_class)[j]}'
        print(root_class)
        val_data.append(root_class)

print(len(val_data))
write_json("val.json", val_data)

