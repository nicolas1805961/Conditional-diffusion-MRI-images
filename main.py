import torchvision
from datasets import load_dataset
from torchvision import transforms
import torch
from dataloader import CustomDataloaderTrain, CustomDataloaderVal
from get_model import get_model, ClassConditionedUnet
from diffusers import DDPMScheduler, DDIMScheduler
from train import train
import os
from datetime import datetime
from copy import copy
from torch.utils.tensorboard import SummaryWriter
from tools import read_config_video
import argparse
from pathlib import Path
import numpy as np
import random
import torch.backends.cudnn as cudnn
from ruamel.yaml import YAML

parser = argparse.ArgumentParser()
parser.add_argument("-config", help="yaml config file", required=True)
parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
args = parser.parse_args()
deterministic = args.deterministic

if deterministic:
    random.seed(12345)
    np.random.seed(12345)
    torch.cuda.manual_seed_all(12345)
    torch.manual_seed(12345)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


output_folder = 'out'

# Or load images from a local folder
data_path = os.path.join('..', 'Isensee_unlabeled', 'nnunet', 'Lib_resampling_training_mask')
#data_path = r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\Lib_resampling_training_mask"
train_dataset = CustomDataloaderTrain(path=data_path)
val_dataset = CustomDataloaderVal(path=data_path)

config = read_config_video(os.path.join(Path.cwd(), args.config))

# You can lower your batch size if you're running out of GPU memory
batch_size = config['batch_size']
batch_size_val = config['batch_size_val']
epochs = config['epochs']
validation_step = config['validation_step']
guidance_scale = config['guidance_scale']
num_train_timesteps = config['num_train_timesteps']
num_val_timesteps = config['num_val_timesteps']

# Create a dataloader from the dataset to serve up the transformed images in batches
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True)

#model = get_model()
model = ClassConditionedUnet()

timestr = datetime.now().strftime("%Y-%m-%d_%HH%M_%Ss_%f")
log_dir = os.path.join(copy(output_folder), timestr)
writer = SummaryWriter(log_dir=log_dir)

yaml = YAML()
with open(os.path.join(log_dir, 'config.yaml'), 'wb') as f:
    yaml.dump(config, f)

# Set the noise scheduler
if config['scheduler'] == 'ddpm':
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="squaredcos_cap_v2")
    noise_scheduler.set_timesteps(num_inference_steps=num_val_timesteps)
elif config['scheduler'] == 'ddim':
    noise_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
    noise_scheduler.set_timesteps(num_inference_steps=num_val_timesteps)

train(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, noise_scheduler=noise_scheduler, writer=writer, epochs=epochs, validation_step=validation_step, guidance_scale=guidance_scale, log_dir=log_dir)