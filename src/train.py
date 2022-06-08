import torch
import torch.nn as nn
from models.build import resnet
from ema import EMA
from losses import LabelSmoothingCrossEntropy, AMSoftmax, FocalLoss, BinaryCrossEntropy, SupConLoss, SoftTargetCrossEntropy
from utils import get_config
from dataset import get_loader, IR_Dataset
from engine import Trainer, training_experiment
from resnet_big import SupConResNet

def train():
	# Get Device
	if torch.cuda.is_available():
		device = 'cuda'
	else:
		device = 'cpu'

	# Get Config
	cfgs = get_config()

	# Setup Model
	model = resnet().to(device)
	# model = SupConResNet(name='resnet34')
	ema_model = EMA(model.parameters(), decay_rate=0.995, num_updates=0)
	criterion = LabelSmoothingCrossEntropy().to(device)
	# criterion = FocalLoss().to(device)
	# criterion = BinaryCrossEntropy().to(device)	
	# criterion = ContrastiveLoss().to(device)
	# criterion = AMSoftmax().to(device)
	# criterion = SoftTargetCrossEntropy().to(device)
	optimizer =torch.optim.AdamW(model.parameters(), lr=cfgs['train']['lr'])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=3, verbose=True)
	train_loader, test_loader = get_loader(cfgs, IR_Dataset)

	# for img, label in train_loader:
		# print(img.shape)
		# break

	# Setup Training 
	trainer = Trainer(model, criterion, optimizer, ema_model, cfgs['train']["loss_ratio"], cfgs['train']["clip_value"], device=device)

	# Start Training
	training_experiment(train_loader, test_loader, trainer, cfgs['train']['epoch_n'], scheduler)
	print("DONE!")