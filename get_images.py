import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import argparse
from tools import read_config_video
from dataloader import CustomDataloaderVal
from get_model import ClassConditionedUnet
from diffusers import DDPMScheduler, DDIMScheduler

parser = argparse.ArgumentParser()
parser.add_argument("-path", help="path where weights are stored", required=True)
args = parser.parse_args()

output_folder = 'out'

config = read_config_video(os.path.join(args.path, 'config.yaml'))

# You can lower your batch size if you're running out of GPU memory
batch_size = config['batch_size']
batch_size_val = config['batch_size_val']
epochs = config['epochs']
validation_step = config['validation_step']
guidance_scale = config['guidance_scale']
num_train_timesteps = config['num_train_timesteps']
num_val_timesteps = config['num_val_timesteps']

# Or load images from a local folder
data_path = r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\Lib_resampling_testing_mask"
val_dataset = CustomDataloaderVal(path=data_path, test=True)

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)

model = ClassConditionedUnet()

model.load_state_dict(torch.load(os.path.join(args.path, 'weights.pth'), weights_only=True))
model.eval()

# Set the noise scheduler
if config['scheduler'] == 'ddpm':
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="squaredcos_cap_v2")
    noise_scheduler.set_timesteps(num_inference_steps=num_val_timesteps)
elif config['scheduler'] == 'ddim':
    noise_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
    noise_scheduler.set_timesteps(num_inference_steps=num_val_timesteps)

for step, (y) in enumerate(val_dataloader):

    y = y.to('cuda:0')

    # Prepare random x to start from, plus some desired labels y
    x = torch.randn(y.shape[0], 1, 192, 192).to('cuda:0')

    y_uncond = torch.zeros_like(y)

    y_new = torch.cat([y_uncond, y])

    timestep_images = []

    # Sampling loop
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

        # Expand the latents if we are doing classifier free guidance
        x_stacked = torch.cat([x] * 2)

        # Get model pred
        with torch.no_grad():
            residual = model(x_stacked, t, y_new)  # Again, note that we pass in our labels y
        
        # Perform guidance
        residual_pred_uncond, residual_pred_label = residual.chunk(2)
        residual = residual_pred_uncond + guidance_scale * (residual_pred_label - residual_pred_uncond)

        # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample

        if i % ((noise_scheduler.timesteps[0] + 1) / 5) == 0:
            print(i)
            timestep_images.append(x)

    timestep_images = torch.stack(timestep_images)
    y = torch.argmax(y, dim=1).cpu().numpy()[:, None]

    fig, ax = plt.subplots(y.shape[0], len(timestep_images) + 2)
    for i in range(y.shape[0]):
        ax[i, 0].imshow(y[i, 0], cmap='gray')
        ax[i, 0].axis('off')
        for j in range(len(timestep_images)):
            ax[i, j+1].imshow(timestep_images[j, i, 0].cpu().numpy(), cmap='gray')
            ax[i, j+1].axis('off')
        ax[i, j+2].imshow(x[i, 0].cpu().numpy(), cmap='gray')
        ax[i, j+2].axis('off')
    ax[i, 0].text(0.5,-0.1, "Input segmentation label", size=5, ha="center", transform=ax[i, 0].transAxes)
    ax[i, j+2].text(0.5,-0.1, "Output generated image", size=5, ha="center", transform=ax[i, j+2].transAxes)
    plt.subplots_adjust(left=0.02, bottom=0.0, right=0.98, top=1.0, wspace=0.1, hspace=-0.6)
    plt.savefig('example.png', dpi=600)
    plt.show()