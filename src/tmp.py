import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning)
import sys
sys.path.append('./models/')
from build import build_model

class Trainer:
	def __init__(self, model, criterion, optimizer, ema_model, loss_ratio=0.1,
				 clip_value=1, ckpt='../weights/model.pth', device='cuda'):
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.loss_ratio = loss_ratio
		self.clip_value = clip_value
		self.device = device
		self.ema_model = ema_model
		self.BEST_LOSS = np.inf
		self.ckpt = ckpt
		self.labels = []
		self.predicts = []
		self.load_weights()

	def load_weights(self):
		try:
			self.model.load_state_dict(torch.load(self.ckpt))
			# self.ema_model.store(self.model.parameters())
			# self.ema_model.copy(self.model.parameters())
			print("SUCCESSFULLY LOAD TRAINED MODELS !")
		except:
			print('FIRST TRAINING >>>')

	def train_epoch(self, train_loader):
		self.model.train()
		train_loss_epoch = 0
		train_acc_epoch = []
		for img, label in tqdm(train_loader):
			self.optimizer.zero_grad()

			# img = torch.cat([img[0], img[1]], dim=0)
			# label = label.to(self.device)
			# img = img.to(self.device)

			# print(img.shape)
			# bsz = label.shape[0]

			feature, out = None, self.model(img)

			# print(out.shape)

			# f1, f2 = torch.split(out, [bsz, bsz], dim=0)
			# out = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

			# print(out.shape)
			# print(label.squeeze(dim=1).shape)
			# print(out)

			loss = self.criterion(out, label.squeeze(dim=1))
			train_loss_epoch += loss.item()
			loss.backward()
			
			# nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)
			self.optimizer.step()
			self.ema_model.update(self.model.parameters())
			print("xxxxxxxxxxxx")
			_, predict = out.max(dim=1)
			print(predict)
			train_acc_epoch.append(accuracy_score(predict.cpu().numpy(), label.cpu().numpy()))
		return sum(train_acc_epoch) / len(train_acc_epoch), train_loss_epoch

	def val_epoch(self, val_loader):
		self.model.eval()
		self.ema_model.store(self.model.parameters())
		self.ema_model.copy(self.model.parameters())
		val_loss_epoch = 0
		val_acc_epoch = []
		with torch.no_grad():
			for img, label in tqdm(val_loader):
				img = img.to(self.device)
				feature, out = None, self.model(img)
				
				label = label.to(self.device)
				loss = self.criterion(out, label.squeeze(dim=1))
				val_loss_epoch += loss.item()

				_, predict = out.max(dim=1)
				self.labels.append(label.squeeze(dim=1).detach().cpu().numpy())
				self.predicts.append(predict.detach().cpu().numpy())
				val_acc_epoch.append(accuracy_score(predict.cpu().numpy(), label.cpu().numpy()))
				self.val_loss_epoch = val_loss_epoch
				self.ema_model.copy_back(self.model.parameters())
			return sum(val_acc_epoch) / len(val_acc_epoch), val_loss_epoch

	def save_checkpoint(self):
		if self.val_loss_epoch < self.BEST_LOSS:
			self.BEST_LOSS = self.val_loss_epoch
			torch.save(self.model.state_dict(),self.ckpt)
			print("LOG CONFUSION MATRIX")


def training_experiment(train_loader, test_loader, trainer, epoch_n, scheduler):
	print("BEGIN TRAINING ...")
	accuracy_training = []
	for epoch in range(epoch_n):
		mean_train_acc, train_loss_epoch = trainer.train_epoch(train_loader)

		mean_val_acc, val_loss_epoch = trainer.val_epoch(test_loader)
		scheduler.step(val_loss_epoch)

		print("EPOCH: ", epoch + 1, " - TRAIN_LOSS: ", train_loss_epoch, " - TRAIN_ACC: ", mean_train_acc,
			  " || VAL_LOSS: ", val_loss_epoch, " - VAL_ACC: ", mean_val_acc)
		accuracy_training.append(mean_val_acc)
	print("MAX_ACC = ",max(accuracy_training))
		

























import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning)
import sys
sys.path.append('./models/')
from build import build_model

class Trainer:
	def __init__(self, model, criterion, optimizer, ema_model, loss_ratio=0.1,
				 clip_value=1, ckpt='../weights/model.pth', device='cuda'):
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.loss_ratio = loss_ratio
		self.clip_value = clip_value
		self.device = device
		self.ema_model = ema_model
		self.BEST_LOSS = np.inf
		self.ckpt = ckpt
		self.labels = []
		self.predicts = []
		self.load_weights()

	def load_weights(self):
		try:
			self.model.load_state_dict(torch.load(self.ckpt))
			# self.ema_model.store(self.model.parameters())
			# self.ema_model.copy(self.model.parameters())
			print("SUCCESSFULLY LOAD TRAINED MODELS !")
		except:
			print('FIRST TRAINING >>>')

	def train_epoch(self, train_loader):
		self.model.train()
		train_loss_epoch = 0
		train_acc_epoch = []
		for img1, label1, img2, label2 in tqdm(train_loader):
			self.optimizer.zero_grad()

			img1 = img1.to(self.device)
			img2 = img2.to(self.device)
			label1 = label1.to(self.device)
			out1 = self.model(img1)
			out2 = self.model(img2)
			loss = self.criterion(out1, out2, label1.squeeze(dim=1))
			train_loss_epoch += loss.item()
			loss.backward()
			
			self.optimizer.step()
			self.ema_model.update(self.model.parameters())

			_, predict = out1.max(dim=1)
			train_acc_epoch.append(accuracy_score(predict.cpu().numpy(), label1.cpu().numpy()))
		return sum(train_acc_epoch) / len(train_acc_epoch), train_loss_epoch

	def val_epoch(self, val_loader):
		self.model.eval()
		self.ema_model.store(self.model.parameters())
		self.ema_model.copy(self.model.parameters())
		val_loss_epoch = 0
		val_acc_epoch = []
		with torch.no_grad():
			for img, label in tqdm(val_loader):
				img = img.to(self.device)
				feature, out = None, self.model(img)
				
				label = label.to(self.device)
				loss = self.criterion(out, label.squeeze(dim=1))
				val_loss_epoch += loss.item()

				_, predict = out.max(dim=1)
				self.labels.append(label.squeeze(dim=1).detach().cpu().numpy())
				self.predicts.append(predict.detach().cpu().numpy())
				val_acc_epoch.append(accuracy_score(predict.cpu().numpy(), label.cpu().numpy()))
				self.val_loss_epoch = val_loss_epoch
				self.ema_model.copy_back(self.model.parameters())
			return sum(val_acc_epoch) / len(val_acc_epoch), val_loss_epoch

	def save_checkpoint(self):
		if self.val_loss_epoch < self.BEST_LOSS:
			self.BEST_LOSS = self.val_loss_epoch
			torch.save(self.model.state_dict(),self.ckpt)
			print("LOG CONFUSION MATRIX")


def training_experiment(train_loader, test_loader, trainer, epoch_n, scheduler):
	print("BEGIN TRAINING ...")
	accuracy_training = []
	for epoch in range(epoch_n):
		mean_train_acc, train_loss_epoch = trainer.train_epoch(train_loader)

		mean_val_acc, val_loss_epoch = trainer.val_epoch(test_loader)
		scheduler.step(val_loss_epoch)

		print("EPOCH: ", epoch + 1, " - TRAIN_LOSS: ", train_loss_epoch, " - TRAIN_ACC: ", mean_train_acc,
			  " || VAL_LOSS: ", val_loss_epoch, " - VAL_ACC: ", mean_val_acc)
		accuracy_training.append(mean_val_acc)
	print("MAX_ACC = ",max(accuracy_training))
		

