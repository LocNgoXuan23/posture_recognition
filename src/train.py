from comet_ml import Experiment
import torch
import torch.nn as nn
from models.build import resnet
from ema import EMA
from losses import LabelSmoothingCrossEntropy 
from utils import get_config
from dataset import get_loader, IR_Dataset
from engine import Trainer, training_experiment


def train():
    # Get Device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Get Config
    cfgs = get_config()

    # Setup Experiment
    with open('../experiment_apikey.txt','r') as f:
        api_key = f.read()
    experiment = Experiment(
        api_key = api_key,
        project_name = "IR Project",
        workspace = "maxph2211",
    )
    experiment.log_parameters(cfgs)

    # Setup Model
    model = resnet().to(device)
    ema_model = EMA(model.parameters(), decay_rate=0.995, num_updates=0)
    criterion = LabelSmoothingCrossEntropy().to(device)
    optimizer =torch.optim.AdamW(model.parameters(), lr=cfgs['train']['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=3, verbose=True)
    train_loader, test_loader = get_loader(cfgs, IR_Dataset)
    print(len(train_loader))
    print(len(test_loader))

    # Setup Training 
    trainer = Trainer(model, criterion, optimizer, ema_model, cfgs['train']["loss_ratio"], cfgs['train']["clip_value"], device=device)

    # Start Training
    training_experiment(train_loader, test_loader, experiment, trainer, cfgs['train']['epoch_n'], 
    scheduler)
    print("DONE!")