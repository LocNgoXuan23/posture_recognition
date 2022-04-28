import torch
from torchvision import  transforms
from utils import read_json, get_label
from torch.utils.data import Dataset, DataLoader
import cv2
import random 


class IR_Dataset(Dataset):
    def __init__(self, cfgs, mode = 'train'):
        self.cfgs = cfgs
        self.mode = mode
        self.data_path = cfgs['data'][mode+'_path']
        self.data = read_json(self.data_path)

    def __getitem__(self, item):
        image_path = self.data[item]
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(image_path)
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        img, label = process(image_path, self.cfgs)
        return img, label

    def __len__(self):
        return len(self.data)

def process(image_path, cfgs):
    label = get_label(image_path)
    img = cv2.imread(image_path)#, cv2.IMREAD_GRAYSCALE)
    thres = random.choice(list(range(0,15)))
    img[img < thres] = 0# cfgs['data']['pix_thres']] = 0 
    # img = cv2.equalizeHist(img)
    img = cv2.resize(img, (224, 224)) #cfgs['data']['image_size'])
    img = transform(img)
    label = torch.LongTensor([int(label)])
    return img, label

def get_loader(cfgs, dataset_type):
  train_dataset = dataset_type(cfgs,'train')
  val_dataset = dataset_type(cfgs,'val')

  train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = cfgs['data']['batch_size'],
        num_workers = cfgs['data']['num_workers']
  )

  val_loader = DataLoader(
          dataset = val_dataset,
          batch_size = cfgs['data']['batch_size'],
          num_workers = cfgs['data']['num_workers']
  )
  print("DONE LOADING DATA !")
  return train_loader, val_loader

def transform(image):
    return transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5,)),
       transforms.RandomRotation(10),
        # transforms.RandomHorizontalFlip(p=0.2),
       transforms.RandomAutocontrast(p=0.2),
        #  transforms.RandomCrop(224),
         ]
    )(image)