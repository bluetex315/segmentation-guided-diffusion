import os
import argparse

# torch imports
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from torchinfo import summary

# HF imports
import diffusers
from diffusers.optimization import get_cosine_schedule_with_warmup
import datasets

# custom imports
from training import train_loop
from eval import evaluate_generation, evaluate_sample_many, evaluate_fake_PIRADS_images, evaluate, SegGuidedDDIMPipeline
from utils import make_grid, save_nifti, load_config, flatten_config, parse_3d_volumes, train_test_split_dset

import yaml
import pickle
import nibabel as nib
from sklearn.model_selection import train_test_split
import monai
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, NormalizeIntensityd, ToTensord, Orientationd, CenterSpatialCropd, Orientationd, ScaleIntensityRangePercentilesd
from datetime import datetime
from tqdm.auto import tqdm


def main(
    config,
    eval_shuffle_dataloader=True,

    # arguments only used in eval
    eval_mask_removal=False,
    eval_blank_mask=False,
    eval_sample_size=1000
):
    print("")
    print("SETTINGS:")
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

    if config['class_conditional']:
        config['output_dir'] += "-classCond"
    
    if config['cfg_training']:
        config['output_dir'] += "-CFG"
    
    if config['mode'] == 'eval':
        config['output_dir'] += "-eval"
        
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
    
    print("main line113", slices_dset_list_train[0].keys())
    
    norm_key, tot_key = [], []
    if config['img_dir'] is not None:
        norm_key.append('image')
        tot_key.append('image')

    if config['segmentation_guided']:
        tot_key.append(seg_key)
    
    if config['neighboring_images_guided']:
        tot_key.append('clean_left')
        tot_key.append('clean_right')
        norm_key.append('clean_left')
        norm_key.append('clean_right')

    train_transforms = Compose([

        # Add a channel dimension to 'image' and 'segm'
        EnsureChannelFirstd(keys=tot_key, channel_dim='no_channel'),

        Orientationd(keys=tot_key, axcodes='LAS'),

        # center spatial crop
        CenterSpatialCropd(keys=tot_key, roi_size=(128, 128)),

        # scale to [0, 1]
        ScaleIntensityRangePercentilesd(
            keys=norm_key,
            lower=2.5, upper=97.5, 
            b_min=0.0, b_max=1.0, 
            clip=True, relative=False, 
            channel_wise=False
        ),
        
        # normalize
        NormalizeIntensityd(
            keys=norm_key,
            subtrahend=0.5,
            divisor=0.5
        ),
        
        # Convert 'image' and 'segm' to PyTorch tensors
        ToTensord(keys=tot_key)
    ])

    eval_transforms = Compose([

        # Add a channel dimension to 'image' and 'segm'
        EnsureChannelFirstd(keys=tot_key, channel_dim='no_channel'),

        Orientationd(keys=tot_key, axcodes='LAS'),
        
        # center spatial crop
        CenterSpatialCropd(keys=tot_key, roi_size=(config['img_size'], config['img_size'])),

        # scale to [0, 1]
        ScaleIntensityRangePercentilesd(
            keys=norm_key,
            lower=1.0, upper=99.0, 
            b_min=0.0, b_max=1.0, 
            clip=True, relative=False, 
            channel_wise=False
        ),
        
        # normalize
        NormalizeIntensityd(
            keys=norm_key,
            subtrahend=0.5,
            divisor=0.5
        ),

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

    if config['mode'] == 'eval':

        if config['use_alldset_for_eval']:
            slices_dset_list_all = slices_dset_list_train + slices_dset_list_val + slices_dset_list_test
            all_dataset_for_eval = monai.data.Dataset(slices_dset_list_all, transform=eval_transforms)

            print(f"Length of test_dataset is {len(all_dataset_for_eval)}")
        
            eval_dataloader = torch.utils.data.DataLoader(
                all_dataset_for_eval, 
                batch_size=config['eval_batch_size'],
                shuffle=False
            )
        
        else:
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
        
        if config['neighboring_images_guided']:     # concat clean_left and clean_right to noisy_images
            in_channels += 2    

        if config['class_conditional']:
            # in_channels += 1
            num_class_embeds = config['num_class_embeds']
        else:
            num_class_embeds = None


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
        class_embed_type=None,
        num_class_embeds=num_class_embeds
    )

    dummy_sample = torch.randn(1, in_channels, config['img_size'], config['img_size'])
    dummy_timestep = torch.tensor([50])
    dummy_class_label = torch.tensor([0])
    
    if config['class_conditional']:
        print("")
        summary(model, input_data=(dummy_sample, dummy_timestep, dummy_class_label))
        print("")
    else:
        print("")
        summary(model, input_data=(dummy_sample, dummy_timestep))
        print("")
        
    if (config['mode'] == "train" and config['resume_epoch'] is not None) or "eval" in config['mode']:
        if config['mode'] == "train":
            print("resuming from model at training epoch {}".format(config['resume_epoch']))
        elif "eval" in config['mode']:
            print("loading saved model...")
        model = model.from_pretrained(config['checkpoint'], use_safetensors=True)

    model = nn.DataParallel(model)
    model.to(device)

    # define noise scheduler
    if config['model_type'] == "DDPM":
        noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)
    elif config['model_type'] == "DDIM":
        if config['use_squaredcos_cap_v2_scheduler']:
            noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000, beta_start=0.00005, beta_end=0.01, beta_schedule="squaredcos_cap_v2")
        else:
            noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000, beta_start=0.00005, beta_end=0.01, beta_schedule="linear")

    if config['mode'] == "train":
        # training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config['lr_warmup_steps'],
            num_training_steps=(len(train_dataloader) * config['num_epochs']),
        )

        # Save the images
        train_save_dir = os.path.join(config['output_dir'], "training", "images_after_transform")
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
            
    elif config['mode'] == "eval":
        """
        default eval behavior:
        evaluate image generation or translation (if for conditional model, either evaluate naive class conditioning but not CFG,
        or with CFG),
        possibly conditioned on masks.

        has various options.
        """
        if config['model_type']== "DDIM":
            if config['segmentation_guided']:
                pipeline = SegGuidedDDIMPipeline(
                    unet=model.module, scheduler=noise_scheduler, eval_dataloader=eval_dataloader, external_config=config
                    )
            else:
                pipeline = diffusers.DDIMPipeline(unet=model.module, scheduler=noise_scheduler)
        else:
            raise NotImplementedError("TODO: implement eval for DDPM")

        epoch = config['num_epochs'] - 1
        if config['segmentation_guided']:
            for seg_batch in tqdm(eval_dataloader):
                if config['fake_labels']:
                    evaluate_fake_PIRADS_images(config, epoch, pipeline, seg_batch)
                else:
                    evaluate(config, epoch, pipeline, seg_batch)        # evaluate only saves synthetic images


    elif config['mode'] == "eval_many":
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
    parser.add_argument("--mode", type=str, default="train", help="train or eval")
    parser.add_argument("--checkpoint", type=str, default="", help='checkpoint to torch safetensor')
    parser.add_argument("--use_alldset_for_eval", action='store_true')
    args = parser.parse_args()

    # Load and flatten the configuration.
    nested_config = load_config(args.config_file)
    flat_config = flatten_config(nested_config)

    cmd_args = vars(args)
    flat_config.update(cmd_args)

    main(flat_config)

