import os
import argparse

# torch imports
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

# HF imports
import diffusers
from diffusers.optimization import get_cosine_schedule_with_warmup
import datasets

# custom imports
from training import TrainingConfig, train_loop
from eval import evaluate_generation, evaluate_sample_many
from utils import make_grid, save_nifti, load_config, flatten_config, parse_3d_volumes, train_test_split_dset

import yaml
import pickle
import nibabel as nib
from sklearn.model_selection import train_test_split
import monai
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, NormalizeIntensityd, ToTensord, Orientationd, CenterSpatialCropd, Orientationd
from datetime import datetime


def main(
    config,
    eval_shuffle_dataloader=True,

    # arguments only used in eval
    eval_mask_removal=False,
    eval_blank_mask=False,
    eval_sample_size=1000
):
    
    print("TRAINING SETTINGS:")
    print(config)
    #GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on {}'.format(device))

    # load config
    config['output_dir'] = '{}-{}-{}'.format(config['model_type'].lower(), config['dataset'], config['img_size'])   # the model namy locally and on the HF Hub

    if config['segmentation_guided']:
        config['output_dir'] += "-segguided"
        assert config['seg_dir'] is not None, "must provide segmentation directory for segmentation guided training/sampling"

    if config['use_ablated_segmentations'] or eval_mask_removal or eval_blank_mask:
        config['output_dir'] += "-ablated"

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['output_dir'] += f"_{timestamp}"
    config['output_dir'] = os.path.join("experiments", config['output_dir'])
    print("output dir: {}".format(config['output_dir']))

    if config['mode'] == "train":
        evalset_name = "val"
        assert config['img_dir'] is not None, "must provide image directory for training"
    elif "eval" in config['mode']:
        evalset_name = "test"

    print("using evaluation set: {}".format(evalset_name))

    load_images_as_np_arrays = False
    if config['num_img_channels'] not in [1, 3]:
        load_images_as_np_arrays = True
        print("image channels not 1 or 3, attempting to load images as np arrays...")

    dset_dict = {}
    if config['img_dir'] is not None:
        img_paths = [
            os.path.join(root, file)
            for root, _, files in sorted(os.walk(config['img_dir']))
            for file in files if file.endswith("T2W.nii.gz")
        ]
        dset_dict["image"] = img_paths
    
    if config['segmentation_guided']:
        seg_types = os.listdir(config['seg_dir'])
        seg_paths = {
            seg_type: [
                os.path.join(root, file)
                for root, _, files in sorted(os.walk(os.path.join(config['seg_dir'], seg_type)))
                for file in files if file.endswith(".nii.gz")
            ]
            for seg_type in seg_types
        }
        # print(seg_paths)
        for seg_type in seg_types:
            print("CONDITIONED ON", seg_type)
        seg_key = 'seg_' + seg_type
        dset_dict.update({seg_key: seg_paths[seg_type] for seg_type in seg_types})

        # if "image" not in dset_dict:
        #     dset_dict["patient_id"] = [os.path.basename(f).split("_")[0] for f in seg_paths[seg_types[0]]]

    print(dset_dict.keys())
    # print(dset_dict['seg_gland_mask'])
    
    # train_test_split on patient level
    dset_dict_train, dset_dict_eval = train_test_split_dset(dset_dict, test_size=0.2)
    dset_dict_val, dset_dict_test = train_test_split_dset(dset_dict_eval, test_size=0.5)

    slices_dset_list_train = parse_3d_volumes(dset_dict_train, seg_type, csv_file=config['label_csv_dir'])
    slices_dset_list_val = parse_3d_volumes(dset_dict_val, seg_type, csv_file=config['label_csv_dir'])
    slices_dset_list_test = parse_3d_volumes(dset_dict_test, seg_type, csv_file=config['label_csv_dir'])
    
    norm_key, tot_key = [], []
    if config['img_dir'] is not None:
        norm_key.append('image')
        tot_key.append('image')
    if config['segmentation_guided']:
        tot_key.append(seg_key)

    train_transforms = Compose([

        # Add a channel dimension to 'image' and 'segm'
        EnsureChannelFirstd(keys=tot_key, channel_dim='no_channel'),

        Orientationd(keys=tot_key, axcodes='LAS'),

        # center spatial crop
        CenterSpatialCropd(keys=tot_key, roi_size=(128, 128)),

        # Normalize the 'image' key - zero mean, unit variance normalization
        NormalizeIntensityd(keys=norm_key),
        
        # Convert 'image' and 'segm' to PyTorch tensors
        ToTensord(keys=tot_key)
    ])

    eval_transforms = Compose([

        # Add a channel dimension to 'image' and 'segm'
        EnsureChannelFirstd(keys=tot_key, channel_dim='no_channel'),

        Orientationd(keys=tot_key, axcodes='LAS'),
        
        # center spatial crop   config
        CenterSpatialCropd(keys=tot_key, roi_size=(config['img_size'], config['img_size'])),

        # Normalize the 'image' key - zero mean, unit variance normalization
        NormalizeIntensityd(keys=norm_key),
        
        # Convert 'image' and 'segm' to PyTorch tensors
        ToTensord(keys=tot_key)
    ])

    train_dataset = monai.data.Dataset(slices_dset_list_train, transform=train_transforms)
    val_dataset = monai.data.Dataset(slices_dset_list_val, transform=eval_transforms)
    test_dataset = monai.data.Dataset(slices_dset_list_test, transform=eval_transforms)
    
    print(f"Length of train_dataset is {len(train_dataset)}")
    print(f"Length of val_dataset is {len(val_dataset)}")
    print(f"Length of test_dataset is {len(test_dataset)}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['train_batch_size'], 
        shuffle=True
    )

    eval_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config['eval_batch_size'],
        shuffle=False
    )

    if config['mode'] == 'test':
        eval_dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=config['eval_batch_size'],
            shuffle=False
        )

    # define the model
    in_channels = config['num_img_channels']
    if config['segmentation_guided']:
        assert config['num_segmentation_classes'] is not None
        assert config['num_segmentation_classes'] > 1, "must have at least 2 segmentation classes (INCLUDING background)" 
        if config['segmentation_channel_mode'] == "single":
            in_channels += 1
        elif config['segmentation_channel_mode'] == "multi":
            in_channels = len(seg_types) + in_channels

    model = diffusers.UNet2DModel(
        sample_size=config['img_size'],  # the target image resolution
        in_channels=in_channels,  # the number of input channels, 3 for RGB images
        out_channels=config['num_img_channels'],  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        ),
    )

    if (config['mode'] == "train" and config['resume_epoch'] is not None) or "eval" in config['mode']:
        if config['mode'] == "train":
            print("resuming from model at training epoch {}".format(config['resume_epoch']))
        elif "eval" in config['mode']:
            print("loading saved model...")
        model = model.from_pretrained(os.path.join(config['training_args']['output_dir'], 'unet'), use_safetensors=True)

    model = nn.DataParallel(model)
    model.to(device)

    # define noise scheduler
    if config['model_type'] == "DDPM":
        noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)
    elif config['model_type'] == "DDIM":
        noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000)

    if config['mode'] == "train":
        # training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config['lr_warmup_steps'],
            num_training_steps=(len(train_dataloader) * config['num_epochs']),
        )

        # Save the images
        train_save_dir = os.path.join(config['output_dir'], "training")
        os.makedirs(train_save_dir, exist_ok=True)

        training_example_batch = next(iter(train_dataloader))
        save_nifti(training_example_batch, seg_key, train_save_dir)
        print(f"\n\nBefore training, save success at {train_save_dir}\n\n")
        # train
        train_loop(
            config, 
            model, 
            noise_scheduler, 
            optimizer, 
            train_dataloader, 
            eval_dataloader, 
            lr_scheduler, 
            device=device
        )
            
    elif mode == "eval":
        """
        default eval behavior:
        evaluate image generation or translation (if for conditional model, either evaluate naive class conditioning but not CFG,
        or with CFG),
        possibly conditioned on masks.

        has various options.
        """
        evaluate_generation(
            config, 
            model, 
            noise_scheduler,
            eval_dataloader, 
            eval_mask_removal=eval_mask_removal,
            eval_blank_mask=eval_blank_mask,
            device=device
        )

    elif mode == "eval_many":
        """
        generate many images and save them to a directory, saved individually
        """
        evaluate_sample_many(
            eval_sample_size,
            config,
            model,
            noise_scheduler,
            eval_dataloader,
            device=device
        )

    else:
        raise ValueError("mode \"{}\" not supported.".format(mode))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()


    # Load and flatten the configuration.
    nested_config = load_config(args.config_file)
    flat_config = flatten_config(nested_config)

    print("Flattened configuration:")
    print(flat_config)
    print("")

    # Construct the TrainingConfig object
    # config = TrainingConfig(
    #     mode=yaml_config['model_args']['mode'],
    #     img_size=yaml_config['data_args']['img_size'],
    #     dataset=yaml_config['data_args']['dataset'],
    #     num_img_channels=yaml_config['data_args']['num_img_channels'],
    #     img_dir=yaml_config['data_args']['img_dir'],
    #     seg_dir=yaml_config['data_args']['seg_dir'],
    #     learning_rate=yaml_config['training_args']['learning_rate'],
    #     save_image_epochs=yaml_config['training_args']['save_image_epochs'],
    #     save_model_epochs=yaml_config['training_args']['save_model_epochs'],
    #     segmentation_guided=yaml_config['data_args']['segmentation_guided'],
    #     segmentation_channel_mode=yaml_config['data_args']['segmentation_channel_mode'],
    #     num_segmentation_classes=yaml_config['data_args']['num_segmentation_classes'],
    #     train_batch_size=yaml_config['training_args']['train_batch_size'],  # Assuming train_batch_size is in data_args
    #     eval_batch_size=yaml_config['training_args']['eval_batch_size'],    # Assuming eval_batch_size is in data_args
    #     num_epochs=yaml_config['training_args']['num_epochs'],  # Accessing directly from yaml_config
    #     output_dir=yaml_config['training_args']['output_dir'],  # Accessing directly
    #     model_type=yaml_config['model_args']['model_type'],
    #     resume_epoch=yaml_config['model_args']['resume_epoch'],
    #     use_ablated_segmentations=yaml_config['training_args']['use_ablated_segmentations']
    # )

    main(flat_config)

