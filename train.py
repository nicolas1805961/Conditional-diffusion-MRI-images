import torchvision
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from inference import generate
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

def train(model, train_dataloader, val_dataloader, noise_scheduler, writer, epochs, validation_step, guidance_scale, log_dir):

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, eta_min=1e-7, T_max=epochs)

    losses = []

    for epoch in range(epochs):
        pbar = tqdm(train_dataloader)
        for step, (x, y) in enumerate(pbar):
            x = x.to('cuda:0')
            y = y.to('cuda:0')
            # Sample noise to add to the images
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (x.shape[0],)).long().to('cuda:0')
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

            r = torch.rand(x.shape[0])

            y[r > 0.85] = torch.zeros_like(y[0])

            #if r > 0.8:
            #    y = torch.zeros_like(y)

            #fig, ax = plt.subplots(1, x.shape[0])
            #for b in range(x.shape[0]):
            #    ax[b].imshow(torch.argmax(y, dim=1)[b].cpu(), cmap='gray')
            #plt.show()

            # Get the model prediction
            pred = model(noisy_x, timesteps, y) # Note that we pass in the labels y

            # Calculate the loss
            loss = F.mse_loss(pred, noise)
            loss.backward()
            losses.append(loss.item())

            writer.add_scalar('Iteration/Training loss', loss.mean(), step * epoch + step)

            # Update the model parameters with the optimizer
            optimizer.step()
            optimizer.zero_grad()
        
        lr_scheduler.step()
        writer.add_scalar('Iteration/Learning rate', optimizer.param_groups[0]['lr'], epoch)

        if (epoch + 1) % validation_step == 0:
            with torch.no_grad():
                loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
                print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")
                max_memory_allocated = torch.cuda.max_memory_allocated(device='cuda:0')
                print("Max GPU Memory allocated:", max_memory_allocated / 10e8, "Gb")
                #overfit_data = {'Train': torch.tensor(self.train_loss).mean().item(), 'Val': torch.tensor(self.val_loss).mean().item()}
                #writer.add_scalars('Epoch/Train_vs_val_loss', overfit_data, epoch)

                generate(noise_scheduler=noise_scheduler, net=model, dataloader=val_dataloader, writer=writer, epoch=epoch, guidance_scale=guidance_scale)
                torch.save(model.state_dict(), os.path.join(log_dir, 'weights.pth'))
                
    torch.save(model.state_dict(), os.path.join(log_dir, 'weights.pth'))