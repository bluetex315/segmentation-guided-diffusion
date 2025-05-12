"""
training utils
"""
from dataclasses import dataclass
import math
import os
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from datetime import timedelta
import imageio
import nibabel as nib

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import torchvision

from torch.utils.tensorboard import SummaryWriter

import diffusers

from eval import evaluate, evaluate_fake_PIRADS_images, add_segmentations_to_noise, add_neighboring_images_to_noise, SegGuidedDDPMPipeline, SegGuidedDDIMPipeline


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, eval_dataloader, lr_scheduler, device='cuda'):
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.

    global_step = 0

    # logging
    run_name = '{}-{}-{}'.format(config['model_type'].lower(), config['dataset'], config['img_size'])
    if config['segmentation_guided']:
        run_name += "-segguided"
    writer = SummaryWriter(comment=run_name)

    # Now you train the model
    start_epoch = 0
    if config['resume_epoch'] is not None:
        start_epoch = config['resume_epoch']

    for epoch in range(start_epoch, config['num_epochs']):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['image']
            clean_images = clean_images.to(device)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            if config['save_forward_process']:
                if epoch == 0 and step == 0:
                    subset_images = clean_images[:4]        # take the first four images
                    patient_id = batch['patient_id'][:4]
                    slice_idx = batch['slice_idx'][:4]
                    print("Training line 106", patient_id, slice_idx)

                    time_steps = [0, 200, 400, 600, 800, 999]

                    forward_outputs = []

                    for t in time_steps:
                        
                        # Create a tensor for the specific timestep (same for all images in the subset)
                        t_tensor = torch.tensor([t] * subset_images.shape[0], device=device).long()
                        # Sample noise for these images
                        noise_subset = torch.randn_like(subset_images)
                        # Add noise using the forward diffusion process at the given timestep
                        noisy_subset = noise_scheduler.add_noise(subset_images, noise_subset, t_tensor)
                        forward_outputs.append(noisy_subset)        # shape [4, C, H, W]

                        if t == 0:
                            nifti_save_dir = os.path.join(config['output_dir'], "training", "noisy_images_forward")
                            os.makedirs(nifti_save_dir, exist_ok=True)

                            # Convert the subset images to a NumPy array.
                            # subset_images is a torch.Tensor of shape [4, C, H, W]
                            noisy_subset_np = noisy_subset.cpu().numpy()  

                            # Save each image as a separate NIfTI file.
                            for i in range(noisy_subset_np.shape[0]):
                                image_np = noisy_subset_np[i]  # shape: [C, H, W]
                                
                                # If the image is grayscale (C==1), remove the channel dimension.
                                if image_np.shape[0] == 1:
                                    image_np = np.squeeze(image_np, axis=0)  # Now shape is [H, W]
                                
                                # Define an identity affine (you can customize this if needed).
                                affine = np.eye(4)
                                
                                # Create a NIfTI image.
                                nifti_image = nib.Nifti1Image(image_np, affine)
                                
                                # Define a save path for this image.
                                nifti_save_path = os.path.join(nifti_save_dir, f"subset_image_{i}_patient{patient_id[i]}_slice{slice_idx[i]}_step{step}.nii.gz")
                                
                                # Save the image.
                                nib.save(nifti_image, nifti_save_path)
                                print(f"Saved subset image {i} as NIfTI at: {nifti_save_path}")

                    forward_outputs_tensor = torch.stack(forward_outputs, dim=0).permute(1, 0, 2, 3, 4)         # shape [4, 6, C, H, W].
                    B, T, C, H, W = forward_outputs_tensor.shape  # B=4, T=6, etc.
                    forward_outputs_tensor = forward_outputs_tensor.reshape(B * T, C, H, W)
                    forward_outputs_tensor = torch.rot90(forward_outputs_tensor, k=1, dims=[2, 3])

                    # Prepare save path and save the grid image using your custom save function.
                    train_save_dir = os.path.join(config['output_dir'], "training", "noisy_images_forward")
                    os.makedirs(train_save_dir, exist_ok=True)
                    train_save_path = os.path.join(train_save_dir, f"noisy_images_epoch{epoch}_step{step}.png")

                    torchvision.utils.save_image(forward_outputs_tensor, train_save_path, nrow=len(time_steps), normalize=True, value_range=(-1, 1))

                    # imageio.imwrite(grid_filename, grid_np_transposed)
                    print(f"Saved grid image for forward process at {train_save_path}")
                                    
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config['num_train_timesteps'], (bs,), device=clean_images.device).long()
            print("training 103 timesteps", timesteps)
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            if config['segmentation_guided']:
                noisy_images = add_segmentations_to_noise(noisy_images, batch, config, device)

            if config['neighboring_images_guided']:
                noisy_images = add_neighboring_images_to_noise(noisy_images, batch, config, device)

            if config['class_conditional']:
                class_labels = batch['class_label'].long().to(device)
                # classifier-free guidance
                if config['cfg_training']:
                    a = np.random.uniform()
                    if a <= config['cfg_p_uncond']:
                        class_labels = torch.zeros_like(class_labels).long()
                noise_pred = model(noisy_images, timesteps, class_labels=class_labels, return_dict=False)[0]
            else:
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            loss = F.mse_loss(noise_pred, noise)
            # print("training 131 loss", loss)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # also train on target domain images if conditional
            # (we don't have masks for this domain, so we can't do segmentation-guided; just use blank masks)
            # if config['class_conditional']:
            #     target_domain_images = batch['images_target']
            #     target_domain_images = target_domain_images.to(device)

            #     # Sample noise to add to the images
            #     noise = torch.randn(target_domain_images.shape).to(target_domain_images.device)
            #     bs = target_domain_images.shape[0]

            #     # Sample a random timestep for each image
            #     timesteps = torch.randint(0, noise_scheduler.config['num_train_timesteps'], (bs,), device=target_domain_images.device).long()

            #     # Add noise to the clean images according to the noise magnitude at each timestep
            #     # (this is the forward diffusion process)
            #     noisy_images = noise_scheduler.add_noise(target_domain_images, noise, timesteps)

            #     if config['segmentation_guided']:
            #         # no masks in target domain so just use blank masks
            #         noisy_images = torch.cat((noisy_images, torch.zeros_like(noisy_images)), dim=1)

            #     # Predict the noise residual
            #     class_labels = torch.full([noisy_images.size(0)], 2).long().to(device)
            #     # classifier-free guidance
            #     a = np.random.uniform()
            #     if a <= config['cfg_p_uncond']:
            #         class_labels = torch.zeros_like(class_labels).long()
            #     noise_pred = model(noisy_images, timesteps, class_labels=class_labels, return_dict=False)[0]
            #     loss_target_domain = F.mse_loss(noise_pred, noise)
            #     loss_target_domain.backward()

            #     nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            #     optimizer.step()
            #     lr_scheduler.step()
            #     optimizer.zero_grad()

            progress_bar.update(1)
            if config['class_conditional']:
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                # logs = {"loss": loss.detach().item(), "loss_target_domain": loss_target_domain.detach().item(), 
                #         "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                writer.add_scalar("loss_target_domain", loss.detach().item(), global_step)
            else: 
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            writer.add_scalar("loss", loss.detach().item(), global_step)

            progress_bar.set_postfix(**logs)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if config['model_type'] == "DDPM":
            if config['segmentation_guided']:
                pipeline = SegGuidedDDPMPipeline(
                    unet=model.module, scheduler=noise_scheduler, eval_dataloader=eval_dataloader, external_config=config
                )
            else:
                if config['class_conditional']:
                    raise NotImplementedError("TODO: Conditional training not implemented for non-seg-guided DDPM")
                else:
                    pipeline = diffusers.DDPMPipeline(unet=model.module, scheduler=noise_scheduler)
        elif config['model_type'] == "DDIM":
            if config['segmentation_guided']:
                pipeline = SegGuidedDDIMPipeline(
                    unet=model.module, scheduler=noise_scheduler, eval_dataloader=eval_dataloader, external_config=config
                )
            else:
                if config['class_conditional']:
                    raise NotImplementedError("TODO: Conditional training not implemented for non-seg-guided DDIM")
                else:
                    pipeline = diffusers.DDIMPipeline(unet=model.module, scheduler=noise_scheduler)

        model.eval()

        if (epoch + 1) % config['save_image_epochs'] == 0 or epoch == config['num_epochs'] - 1:
            # print("output dir ", config['output_dir'])
            if config['segmentation_guided']:
                for seg_batch in tqdm(eval_dataloader):
                    if config['fake_labels']:
                        evaluate_fake_PIRADS_images(config, epoch, pipeline, seg_batch)
                    else:
                        evaluate(config, epoch, pipeline, seg_batch)        # evaluate only saves synthetic images
            else:
                evaluate(config, epoch, pipeline)

        if (epoch + 1) % config['save_model_epochs'] == 0 or epoch == config['num_epochs'] - 1:
            pipeline.save_pretrained(config['output_dir'])
