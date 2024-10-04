import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def generate(noise_scheduler, net, dataloader, writer, epoch, guidance_scale):

    y = next(iter(dataloader))
    y = y.to('cuda:0')

    # Prepare random x to start from, plus some desired labels y
    x = torch.randn(y.shape[0], 1, 192, 192).to('cuda:0')

    y_uncond = torch.zeros_like(y)

    y_new = torch.cat([y_uncond, y])

    # Sampling loop
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

        # Expand the latents if we are doing classifier free guidance
        x_stacked = torch.cat([x] * 2)

        # Get model pred
        with torch.no_grad():
            residual = net(x_stacked, t, y_new)  # Again, note that we pass in our labels y
        
        # Perform guidance
        residual_pred_uncond, residual_pred_label = residual.chunk(2)
        residual = residual_pred_uncond + guidance_scale * (residual_pred_label - residual_pred_uncond)

        # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample

    x = cv.normalize(x.cpu().numpy(), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F).astype(np.uint8)
    writer.add_images('Images', x, epoch, dataformats='NCHW')

    y = torch.argmax(y, dim=1).cpu().numpy()[:, None]
    y = cv.normalize(y, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F).astype(np.uint8)
    writer.add_images('Labels', y, epoch, dataformats='NCHW')