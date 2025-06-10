"""
model evaluation/sampling
"""
import math
import os
import torch
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from copy import deepcopy
import numpy as np

import diffusers
from diffusers import DiffusionPipeline, ImagePipelineOutput, DDIMScheduler, RePaintScheduler
from diffusers.utils.torch_utils import randn_tensor 

from utils import make_grid
from repaint_scheduler import get_schedule_jump, get_schedule_jump_paper
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib

####################
# segmentation-guided DDPM
####################

def evaluate_sample_many(
    sample_size,
    config,
    model,
    noise_scheduler,
    eval_dataloader,
    device='cuda'
    ):

    # for loading segs to condition on:
    # setup for sampling
    if config['model_type']== "DDPM":
        if config['segmentation_guided']:
            pipeline = SegGuidedDDPMPipeline(
                unet=model.module, scheduler=noise_scheduler, eval_dataloader=eval_dataloader, external_config=config
                )
        else:
            pipeline = diffusers.DDPMPipeline(unet=model.module, scheduler=noise_scheduler)
    elif config['model_type']== "DDIM":
        if config['segmentation_guided']:
            pipeline = SegGuidedDDIMPipeline(
                unet=model.module, scheduler=noise_scheduler, eval_dataloader=eval_dataloader, external_config=config
                )
        else:
            pipeline = diffusers.DDIMPipeline(unet=model.module, scheduler=noise_scheduler)


    sample_dir = test_dir = os.path.join(config['output_dir'], "samples_many_{}".format(sample_size))
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    num_sampled = 0
    # keep sampling images until we have enough
    for bidx, seg_batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        if num_sampled < sample_size:
            if config['segmentation_guided']:
                current_batch_size = [v for k, v in seg_batch.items() if k.startswith("seg_")][0].shape[0]
            else:
                current_batch_size = config['eval_batch_size']

            if config['segmentation_guided']:
                images = pipeline(
                    batch_size = current_batch_size,
                    seg_batch=seg_batch,
                ).images
            else:
                images = pipeline(
                    batch_size = current_batch_size,
                ).images

            # save each image in the list separately
            for i, img in enumerate(images):
                if config['segmentation_guided']:
                    # name base on input mask fname
                    img_fname = "{}/condon_{}".format(sample_dir, seg_batch["image_filenames"][i])
                else:
                    img_fname = f"{sample_dir}/{num_sampled + i:04d}.png"
                img.save(img_fname)

            num_sampled += len(images)
            print("sampled {}/{}.".format(num_sampled, sample_size))



def evaluate_generation(
        config, 
        model, 
        noise_scheduler, 
        eval_dataloader, 
        class_label_cfg=None, 
        translate=False, 
        eval_mask_removal=False, 
        eval_blank_mask=False,
        device='cuda'
        ):
    """
    general function to evaluate (possibly mask-guided) trained image generation model in useful ways.
    also has option to use CFG for class-conditioned sampling (otherwise, class-conditional models will be evaluated using naive class conditioning and sampling from both classes).

    can also evaluate for image translation.
    """

    # for loading segs to condition on:
    eval_dataloader = iter(eval_dataloader)

    if config['segmentation_guided']:
        seg_batch = next(eval_dataloader)
        if eval_blank_mask:
            # use blank masks
            for k, v in seg_batch.items():
                if k.startswith("seg_"):
                    seg_batch[k] = torch.zeros_like(v)

    # setup for sampling
    # After each epoch you optionally sample some demo images with evaluate() and save the model
    if config['model_type']== "DDPM":
        if config['segmentation_guided']:
            pipeline = SegGuidedDDPMPipeline(
                unet=model.module, scheduler=noise_scheduler, eval_dataloader=eval_dataloader, external_config=config
                )
        else:
            pipeline = diffusers.DDPMPipeline(unet=model.module, scheduler=noise_scheduler)
    elif config['model_type']== "DDIM":
        if config['segmentation_guided']:
            pipeline = SegGuidedDDIMPipeline(
                unet=model.module, scheduler=noise_scheduler, eval_dataloader=eval_dataloader, external_config=config
                )
        else:
            pipeline = diffusers.DDIMPipeline(unet=model.module, scheduler=noise_scheduler)

    # sample some images
    if config['segmentation_guided']:
        evaluate(config, -1, pipeline, seg_batch, class_label_cfg, translate)
    else:
        if config['class_conditional']:
            raise NotImplementedError("TODO: implement CFG and naive conditioning sampling for non-seg-guided pipelines, including for image translation")
        evaluate(config, -1, pipeline)

    # seg-guided specific visualizations
    if config['segmentation_guided'] and eval_mask_removal:
        plot_result_masks_multiclass = True
        if plot_result_masks_multiclass:
            pipeoutput_type = 'np'
        else:
            pipeoutput_type = 'pil'

        # visualize segmentation-guided sampling by seeing what happens 
        # when segs removed
        num_viz = config['eval_batch_size']

        # choose one seg to sample from; duplicate it
        eval_same_image = False
        if eval_same_image:
            seg_batch = {k: torch.cat(num_viz*[v[:1]]) for k, v in seg_batch.items()}

        result_masks = torch.Tensor()
        multiclass_masks = []
        result_imgs = []
        multiclass_masks_shape = (config['eval_batch_size'], 1, config['img_size'], config['img_size'])

        # will plot segs + sampled images
        for seg_type in seg_batch.keys():
            if seg_type.startswith("seg_"):
                #convert from tensor to PIL
                seg_batch_plt = seg_batch[seg_type].cpu()
                result_masks = torch.cat((result_masks, seg_batch_plt))
        
        # sample given all segs
        multiclass_masks.append(convert_segbatch_to_multiclass(multiclass_masks_shape, seg_batch, config, device))
        full_seg_imgs = pipeline(
            batch_size = num_viz,
            seg_batch=seg_batch,
            class_label_cfg=class_label_cfg,
            translate=translate,
            output_type=pipeoutput_type
        ).images
        if plot_result_masks_multiclass:
            result_imgs.append(full_seg_imgs)
        else:
            result_imgs += full_seg_imgs

        # only sample from masks with chosen classes removed
        chosen_class_combinations = None
        #chosen_class_combinations = [ #for example:
        #   {"seg_all": [1, 2]}
        #]
        if chosen_class_combinations is not None:
            for allseg_classes in chosen_class_combinations:
                # remove all chosen classes
                seg_batch_removed = deepcopy(seg_batch)
                for seg_type in seg_batch_removed.keys():
                    # some datasets have multiple tissue segs stored in multiple masks
                    if seg_type.startswith("seg_"):
                        classes = allseg_classes[seg_type]
                        for mask_val in classes:
                            if mask_val != 0:
                                remove_mask = (seg_batch_removed[seg_type]*255).int() == mask_val
                                seg_batch_removed[seg_type][remove_mask] = 0

                seg_batch_removed_plt = torch.cat([seg_batch_removed[seg_type].cpu() for seg_type in seg_batch_removed.keys() if seg_type.startswith("seg_")])
                result_masks = torch.cat((result_masks, seg_batch_removed_plt))

                multiclass_masks.append(convert_segbatch_to_multiclass(
                multiclass_masks_shape, 
                    seg_batch_removed, config, device))
                # add images conditioned on some segs but not all
                removed_seg_imgs = pipeline(
                    batch_size = config['eval_batch_size'],
                    seg_batch=seg_batch_removed,
                    class_label_cfg=class_label_cfg,
                    translate=translate,
                    output_type=pipeoutput_type
                ).images

                if plot_result_masks_multiclass:
                    result_imgs.append(removed_seg_imgs)
                else:
                    result_imgs += removed_seg_imgs


        else:
            for seg_type in seg_batch.keys():
                # some datasets have multiple tissue segs stored in multiple masks
                if seg_type.startswith("seg_"):
                    seg_batch_removed = seg_batch
                    for mask_val in seg_batch[seg_type].unique():
                        if mask_val != 0:
                            remove_mask = seg_batch[seg_type] == mask_val
                            seg_batch_removed[seg_type][remove_mask] = 0

                            seg_batch_removed_plt = torch.cat([seg_batch_removed[seg_type].cpu() for seg_type in seg_batch.keys() if seg_type.startswith("seg_")])
                            result_masks = torch.cat((result_masks, seg_batch_removed_plt))

                            multiclass_masks.append(convert_segbatch_to_multiclass(
                            multiclass_masks_shape, 
                                seg_batch_removed, config, device))
                            # add images conditioned on some segs but not all
                            removed_seg_imgs = pipeline(
                                batch_size = config['eval_batch_size'],
                                seg_batch=seg_batch_removed,
                                class_label_cfg=class_label_cfg,
                                translate=translate,
                                output_type=pipeoutput_type
                            ).images

                            if plot_result_masks_multiclass:
                                result_imgs.append(removed_seg_imgs)
                            else:
                                result_imgs += removed_seg_imgs

        if plot_result_masks_multiclass:
            multiclass_masks = np.squeeze(torch.cat(multiclass_masks).cpu().numpy())
            multiclass_masks = (multiclass_masks*255).astype(np.uint8)
            result_imgs = np.squeeze(np.concatenate(np.array(result_imgs), axis=0))

            # reverse interleave
            plot_imgs = np.zeros_like(result_imgs)
            plot_imgs[0:len(plot_imgs)//2] = result_imgs[0::2]
            plot_imgs[len(plot_imgs)//2:] = result_imgs[1::2]

            plot_masks = np.zeros_like(multiclass_masks)
            plot_masks[0:len(plot_masks)//2] = multiclass_masks[0::2]
            plot_masks[len(plot_masks)//2:] = multiclass_masks[1::2]

            fig, axs = plt.subplots(
                2, len(plot_masks), 
                figsize=(len(plot_masks), 2), 
                dpi=600
            )

            for i, img in enumerate(plot_imgs):
                if config['data_set'] == 'breast_mri':
                    colors = ['black', 'white', 'red', 'blue']
                elif config['data_set'] == 'ct_organ_large':
                    colors = ['black', 'blue', 'green', 'red', 'yellow', 'magenta']
                elif config['data_set'] == 'fastMRI_NYU':
                    coloc = ['black', 'white']
                else: 
                    raise ValueError('Unknown dataset')

                cmap = ListedColormap(colors)
                axs[0,i].imshow(plot_masks[i], cmap=cmap, vmin=0, vmax=len(colors)-1)
                axs[0,i].axis('off')
                axs[1,i].imshow(img, cmap='gray')
                axs[1,i].axis('off')

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig('ablated_samples_{}.pdf'.format(config['data_set']), bbox_inches='tight')
            plt.show()

        else:
            # Make a grid out of the images
            cols = num_viz
            rows = math.ceil(len(result_imgs) / cols)
            image_grid = make_grid(result_imgs, rows=rows, cols=cols)

            # Save the images
            test_dir = os.path.join(config['output_dir'], "samples")
            os.makedirs(test_dir, exist_ok=True)
            image_grid.save(f"{test_dir}/mask_removal_imgs.png")

            save_image(result_masks, f"{test_dir}/mask_removal_masks.png", normalize=True, 
                    nrow=cols*len(seg_batch.keys()) - 2)

def convert_segbatch_to_multiclass(shape, segmentations_batch, config, device):
    # NOTE: this generic function assumes that segs don't overlap
    # put all segs on same channel
    segs = torch.zeros(shape, dtype=torch.float32).to(device)
    for k, seg in segmentations_batch.items():
        if k.startswith("seg_"):
            seg = seg.to(device, dtype=torch.float32)
            segs[segs == 0] = seg[segs == 0]

    if config['use_ablated_segmentations']:
        # randomly remove class labels from segs with some probability
        segs = ablate_masks(segs, config)

    return segs

def ablate_masks(segs, config, method="equal_weighted"):
    # randomly remove class label(s) from segs with some probability 
    if method == "equal_weighted":
        """
        # give equal probability to each possible combination of removing non-background classes
        # NOTE: requires that each class has a value in ({0, 1, 2, ...} / 255)
        # which is by default if the mask file was saved as {0, 1, 2 ,...} and then normalized by default to [0, 1] by transforms.ToTensor()
        # num_segmentation_classes
        """
        class_removals = (torch.rand(config['num_segmentation_classes'] - 1) < 0.5).int().bool().tolist()
        for class_idx, remove_class in enumerate(class_removals):
            if remove_class:
                segs[(255 * segs).int() == class_idx + 1] = 0

    elif method == "by_class":
        class_ablation_prob = 0.3
        for seg_value in segs.unique():
            if seg_value != 0:
                # remove seg with some probability
                if torch.rand(1).item() < class_ablation_prob:
                    segs[segs == seg_value] = 0
    
    else:
        raise NotImplementedError
    return segs

def add_segmentations_to_noise(noisy_images, segmentations_batch, config, device):
    """
    concat segmentations to noisy image
    """

    if config['segmentation_channel_mode'] == "single":
        multiclass_masks_shape = (noisy_images.shape[0], 1, noisy_images.shape[2], noisy_images.shape[3])
        segs = convert_segbatch_to_multiclass(multiclass_masks_shape, segmentations_batch, config, device) 
        # concat segs to noise
        noisy_images = torch.cat((noisy_images, segs), dim=1)
        
    elif config['segmentation_channel_mode'] == "multi":
        raise NotImplementedError

    return noisy_images

def add_neighboring_images_to_noise(noisy_images, batch, config, device):
    """
    Concatenate neighboring images (e.g., left and right slices) to the noisy image.
    
    Args:
        noisy_images (torch.Tensor): The noisy central images, shape (B, C, H, W).
        batch (dict): A dictionary containing the neighboring images with keys: "clean_left" and "clean_right", each of shape (B, C, H, W).
        config (dict): Configuration parameters, must include 'neighboring_image_channel_mode'.
        device (torch.device): The device to which tensors are moved.
    
    Returns:
        torch.Tensor: The noisy_images tensor with the neighboring images concatenated along the channel dimension.
    """

    left_images = batch["clean_left"].to(device, dtype=noisy_images.dtype)
    right_images = batch["clean_right"].to(device, dtype=noisy_images.dtype)
    noisy_images = torch.cat((noisy_images, left_images, right_images), dim=1)
    
    return noisy_images

def add_adc_to_noise(noisy_images, batch, config, device):
    
    """
    Concatenate adc slice to the noisy image.
    
    Args:
        noisy_images (torch.Tensor): The noisy central images, shape (B, C, H, W).
        batch (dict): A dictionary containing the neighboring images with keys: "clean_left" and "clean_right", each of shape (B, C, H, W).
        config (dict): Configuration parameters, must include 'neighboring_image_channel_mode'.
        device (torch.device): The device to which tensors are moved.
    
    Returns:
        torch.Tensor: The noisy_images tensor with the neighboring images concatenated along the channel dimension.
    """

    adc = batch["adc"].to(device, dtype=noisy_images.dtype)
    noisy_images = torch.cat((noisy_images, adc), dim=1)
    
    return noisy_images


####################
# general DDPM
####################
def evaluate(config, epoch, pipeline, seg_batch=None, class_label_cfg=None, translate=False):
    # Either generate or translate images,
    # possibly mask guided and/or class conditioned.
    # The default pipeline output type is `List[PIL.Image]`     # modified on Dec 12 to be `List[np.ndarray]`

    if config['segmentation_guided']:
        images = pipeline(
            batch_size = len(seg_batch['image']),
            seg_batch=seg_batch,
            class_label_cfg=class_label_cfg,
            translate=translate
        ).images
    else:
        images = pipeline(
            batch_size = len(seg_batch['image']),
            # TODO: implement CFG and naive conditioning sampling for non-seg-guided pipelines (also needed for translation)
        ).images

    # Make a grid out of the images
    cols = 4
    rows = math.ceil(len(images) / cols)

    # Save the images
    test_dir = os.path.join(config['output_dir'], "samples")
    os.makedirs(test_dir, exist_ok=True)

    epoch_test_dir = os.path.join(test_dir, f"{epoch:04d}")
    os.makedirs(epoch_test_dir, exist_ok=True)
    
    # save segmentations we conditioned the samples on segm
    if config['segmentation_guided']:

        print("eval line403 seg_batch keys", seg_batch.keys())
        print("eval line404 seg_batch patient_ids", seg_batch['patient_id'])
        print("eval line405 seg_batch slice_idx", seg_batch['slice_idx'])

        if config['class_conditional']:
            print("eval line406 seg_batch class labels", seg_batch['class_label'])
            # Get the class label for the current sample; ensure it's converted to string if needed.
      
        # save images
        for i in range(len(images)):        # images: List
            pid = seg_batch['patient_id'][i]
            sidx = seg_batch['slice_idx'][i]
            if config['class_conditional']:
                clabel = seg_batch['class_label'][i]
                # You can adjust the formatting as necessary.
                clabel_str = f"_class_{clabel}"
            else:
                clabel_str = ""

            # Create a folder for the current patient if it doesn't exist
            patient_dir = os.path.join(epoch_test_dir, f"patient_{pid}")
            os.makedirs(patient_dir, exist_ok=True)
            
            # save synthetic images
            syn_img_np = images[i, ..., 0]      # np.ndarray
            nifti_syn_img = nib.Nifti1Image(syn_img_np, affine=np.eye(4))
            syn_img_filename = os.path.join(patient_dir, f"{epoch:04d}_{pid}_slice_{sidx}{clabel_str}_syn.nii.gz")
            nib.save(nifti_syn_img, syn_img_filename)

            # save original images
            orig_img_np = seg_batch['image'][i, 0, ...].cpu().numpy()  # Move to CPU if necessary and convert to NumPy
            nifti_orig_img = nib.Nifti1Image(orig_img_np, affine=np.eye(4))
            orig_img_filename = os.path.join(patient_dir, f"{epoch:04d}_{pid}_slice_{sidx}{clabel_str}_orig.nii.gz")
            nib.save(nifti_orig_img, orig_img_filename)

            # save segmentation mask, support multiple masks
            for seg_type in seg_batch.keys():
                if seg_type.startswith("seg_"):
                    segm_np = seg_batch[seg_type][i, 0, ...].cpu().numpy()  # Move to CPU if necessary and convert to NumPy
                    nifti_segm = nib.Nifti1Image(segm_np, affine=np.eye(4))
                    segm_filename = os.path.join(patient_dir, f"{epoch:04d}_{pid}_slice_{sidx}{clabel_str}_cond_segm_{seg_type}.nii.gz")
                    nib.save(nifti_segm, segm_filename)
                
        print("<<<<<<<<<<<<<<<<<<< saved original image at inference >>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
    else:
        for i in range(len(images)):        # images: List
            syn_img_np = np.array(images[i])   # np.ndarray
            nifti_syn_img = nib.Nifti1Image(syn_img_np, affine=np.eye(4))
            syn_img_filename = os.path.join(test_dir, f"{epoch:04d}_{i:04d}_syn.nii.gz")
            nib.save(nifti_syn_img, syn_img_filename)


def evaluate_fake_PIRADS_images(config, epoch, pipeline, seg_batch=None, translate=False):
    """
    Evaluate by generating images conditioned on each of the 5 PI-RADS classes,
    regardless of the true class labels in the segmentation batch.
    """
    
    # Save the images
    test_dir = os.path.join(config['output_dir'], "samples")
    os.makedirs(test_dir, exist_ok=True)

    epoch_test_dir = os.path.join(test_dir, f"{epoch:04d}")
    os.makedirs(epoch_test_dir, exist_ok=True)

    print("[eval] output_dir: ", epoch_test_dir)
    true_class_labels = seg_batch['class_label']

    # Loop over the 5 classes. Adjust this range according to your label convention.
    for fake_class_value in range(5):  # Assuming PI-RADS classes are 1,2,3,4,5
        print(f"<<<<<<<<<<<<<<<<<<< Generating images conditioned on fake class label: {fake_class_value} <<<<<<<<<<<<<<<<<<")

        fake_class_labels = torch.full_like(true_class_labels, fake_class_value)
        print("eval line496", fake_class_labels)
        # Call the pipeline with the fake (override) class label.
        # Your modified __call__ method will detect that fake_class_labels is provided and use it.
        output = pipeline(
            batch_size=len(seg_batch['image']),
            seg_batch=seg_batch,
            fake_class_labels=fake_class_labels,
            translate=translate
        )
        images = output.images  # List or np.array of generated images

        print("eval line508 seg_batch keys", seg_batch.keys())
        print("eval line509 seg_batch patient_ids", seg_batch['patient_id'])
        print("eval line510 seg_batch slice_idx", seg_batch['slice_idx'])
        print("eval line510 seg_batch true_class", seg_batch['class_label'])

        # save images
        for i in range(len(images)):        # images: List
            pid = seg_batch['patient_id'][i]
            sidx = seg_batch['slice_idx'][i]
            if config['class_conditional']:
                true_class_label = true_class_labels[i]
                true_class_label_str = f"_class{true_class_label}"

                fake_class_label = fake_class_value
                fake_class_label_str = f"_fakeclass{fake_class_value}"
            else:
                true_class_label_str = ""
                fake_class_label_str = ""

            # Create a folder for the current patient if it doesn't exist
            patient_dir = os.path.join(epoch_test_dir, f"patient_{pid}")
            os.makedirs(patient_dir, exist_ok=True)
            
            # save synthetic images
            syn_img_np = images[i, ..., 0]      # np.ndarray
            nifti_syn_img = nib.Nifti1Image(syn_img_np, affine=np.eye(4))
            syn_img_filename = os.path.join(patient_dir, f"{epoch:04d}_{pid}_slice{sidx}{fake_class_label_str}_syn.nii.gz")
            nib.save(nifti_syn_img, syn_img_filename)

            # save original images
            orig_img_np = seg_batch['image'][i, 0, ...].cpu().numpy()  # Move to CPU if necessary and convert to NumPy
            nifti_orig_img = nib.Nifti1Image(orig_img_np, affine=np.eye(4))
            orig_img_filename = os.path.join(patient_dir, f"{epoch:04d}_{pid}_slice{sidx}{true_class_label_str}_orig.nii.gz")
            nib.save(nifti_orig_img, orig_img_filename)

            # save segmentation mask, support multiple masks
            for seg_type in seg_batch.keys():
                if seg_type.startswith("seg_"):
                    segm_np = seg_batch[seg_type][i, 0, ...].cpu().numpy()  # Move to CPU if necessary and convert to NumPy
                    nifti_segm = nib.Nifti1Image(segm_np, affine=np.eye(4))
                    segm_filename = os.path.join(patient_dir, f"{epoch:04d}_{pid}_slice{sidx}{true_class_label_str}_cond_segm_{seg_type}.nii.gz")
                    nib.save(nifti_segm, segm_filename)
                
        print("<<<<<<<<<<<<<<<<<<< saved fake images at inference >>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n\n")

class ConfigWrapper:
    def __init__(self, config):
        self.config = config

# custom diffusers pipelines for sampling from segmentation-guided models
class SegGuidedDDPMPipeline(DiffusionPipeline):
    r"""
    Pipeline for segmentation-guided image generation, modified from DDPMPipeline.
    generates both-class conditioned and unconditional images if using class-conditional model without CFG, or just generates 
    conditional images guided by CFG.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
        eval_dataloader ([`torch.utils.data.DataLoader`]):
            Dataloader to load the evaluation dataset of images and their segmentations. Here only uses the segmentations to generate images.
    """
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, eval_dataloader, external_config):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.eval_dataloader = eval_dataloader
        self.external_config = external_config # config is already a thing

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        seg_batch: Optional[torch.Tensor] = None,
        class_label_cfg: Optional[int] = None,
        translate = False,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            seg_batch (`torch.Tensor`, *optional*, defaults to None):
                batch of segmentations to condition generation on
            class_label_cfg (`int`, *optional*, defaults to `None`):
                class label to condition generation on using CFG, if using class-conditional model

            OPTIONS FOR IMAGE TRANSLATION:
            translate (`bool`, *optional*, defaults to False):
                whether to translate images from the source domain to the target domain

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        if self.external_config['segmentation_channel_mode'] == "single":
            img_channel_ct = self.unet.config.in_channels - 1
        elif self.external_config['segmentation_channel_mode'] == "multi":
            img_channel_ct = self.unet.config.in_channels - len([k for k in seg_batch.keys() if k.startswith("seg_")])

        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                img_channel_ct,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            if self.external_config['segmentation_channel_mode'] == "single":
                image_shape = (batch_size, self.unet.config.in_channels - 1, *self.unet.config.sample_size)
            elif self.external_config['segmentation_channel_mode'] == "multi":
                image_shape = (batch_size, self.unet.config.in_channels - len([k for k in seg_batch.keys() if k.startswith("seg_")]), *self.unet.config.sample_size)
            

        # initiate latent variable to sample from
        if not translate:
            # normal sampling; start from noise
            if self.device.type == "mps":
                # randn does not work reproducibly on mps
                image = randn_tensor(image_shape, generator=generator)
                image = image.to(self.device)
            else:
                image = randn_tensor(image_shape, generator=generator, device=self.device)
        else:
            # image translation sampling; start from source domain images, add noise up to certain step, then being there for denoising
            trans_start_t = int(self.external_config['trans_noise_level'] * self.scheduler.config.num_train_timesteps)

            trans_start_images = seg_batch["images"]

            # Sample noise to add to the images
            noise = torch.randn(trans_start_images.shape).to(trans_start_images.device)
            timesteps = torch.full(
                (trans_start_images.size(0),),
                trans_start_t, 
                device=trans_start_images.device
                ).long()
            image = self.scheduler.add_noise(trans_start_images, noise, timesteps)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            if translate:
                # if doing translation, start at chosen time step given partially-noised image
                # skip all earlier time steps (with higher t)
                if t >= trans_start_t:
                  continue

            # 1. predict noise model_output
            # first, concat segmentations to noise
            image = add_segmentations_to_noise(image, seg_batch, self.external_config, self.device)

            if self.external_config['class_conditional']:
                if class_label_cfg is not None:
                    class_labels = torch.full([image.size(0)], class_label_cfg).long().to(self.device)
                    model_output_cond = self.unet(image, t, class_labels=class_labels).sample
                    if self.external_config['use_cfg_for_eval_conditioning']:
                        # use classifier-free guidance for sampling from the given class

                        if self.external_config['cfg_maskguidance_condmodel_only']:
                            image_emptymask = torch.cat((image[:, :img_channel_ct, :, :], torch.zeros_like(image[:, img_channel_ct:, :, :])), dim=1)
                            model_output_uncond = self.unet(image_emptymask, t, 
                                    class_labels=torch.zeros_like(class_labels).long()).sample
                        else:
                            model_output_uncond = self.unet(image, t, 
                                    class_labels=torch.zeros_like(class_labels).long()).sample

                        # use cfg equation
                        model_output = (1. + self.external_config['cfg_weight']) * model_output_cond - self.external_config['cfg_weight'] * model_output_uncond
                    else:
                        # just use normal conditioning
                        model_output = model_output_cond
               
                else:
                    # or, just use basic network conditioning to sample from both classes
                    if self.external_config['class_conditional']:
                        # if training conditionally, evaluate source domain samples
                        class_labels = torch.ones(image.size(0)).long().to(self.device)
                        model_output = self.unet(image, t, class_labels=class_labels).sample
            else:
                model_output = self.unet(image, t).sample
            # output is slightly denoised image

            # 2. compute previous image: x_t -> x_t-1
            # but first, we're only adding denoising the image channel (not seg channel),
            # so remove segs
            image = image[:, :img_channel_ct, :, :]
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        # if training conditionally, also evaluate for target domain images
        # if not using chosen class for CFG
        if self.external_config['class_conditional'] and class_label_cfg is None:
            image_target_domain = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)

            # set step values
            self.scheduler.set_timesteps(num_inference_steps)

            for t in self.progress_bar(self.scheduler.timesteps):
                # 1. predict noise model_output
                # first, concat segmentations to noise
                # no masks in target domain so just use blank masks
                image_target_domain = torch.cat((image_target_domain, torch.zeros_like(image_target_domain)), dim=1)

                if self.external_config['class_conditional']:
                    # if training conditionally, also evaluate unconditional model and target domain (no masks)
                    class_labels = torch.cat([torch.full([image_target_domain.size(0) // 2], 2), torch.zeros(image_target_domain.size(0)) // 2]).long().to(self.device)
                    model_output = self.unet(image_target_domain, t, class_labels=class_labels).sample
                else:
                    model_output = self.unet(image_target_domain, t).sample

                # 2. predict previous mean of image x_t-1 and add variance depending on eta
                # eta corresponds to η in paper and should be between [0, 1]
                # do x_t -> x_t-1
                # but first, we're only adding denoising the image channel (not seg channel),
                # so remove segs
                image_target_domain = image_target_domain[:, :img_channel_ct, :, :]
                image_target_domain = self.scheduler.step(
                    model_output, t, image_target_domain, generator=generator
                ).prev_sample

            image = torch.cat((image, image_target_domain), dim=0)
            # will output source domain images first, then target domain images

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

class SegGuidedDDIMPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation, modified for seg-guided image gen.
    modified from diffusers.DDIMPipeline.
    generates both-class conditioned and unconditional images if using class-conditional model without CFG, or just generates 
    conditional images guided by CFG.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
        eval_dataloader ([`torch.utils.data.DataLoader`]):
            Dataloader to load the evaluation dataset of images and their segmentations. Here only uses the segmentations to generate images.
    
    """
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, eval_dataloader, external_config):
        super().__init__()
        # Wrap the external_config so it behaves like a module
        # wrapped_external_config = SimpleNamespace(**external_config)
        wrapped_external_config = ConfigWrapper(external_config)
        self.register_modules(unet=unet, scheduler=scheduler, eval_dataloader=eval_dataloader, external_config=wrapped_external_config)
        # ^ some reason necessary for DDIM but not DDPM.

        self.eval_dataloader = eval_dataloader
        self.external_config = wrapped_external_config # config is already a thing

        # make sure scheduler can always be converted to DDIM
        scheduler = DDIMScheduler.from_config(scheduler.config)


    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "np.array",
        return_dict: bool = True,
        seg_batch: Optional[torch.Tensor] = None,
        # class_label_cfg: Optional[int] = None,
        fake_class_labels: Optional[torch.Tensor] = None,
        translate = False,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers. A value of `0` corresponds to
                DDIM and `1` corresponds to DDPM.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                If `True` or `False`, see documentation for [`DDIMScheduler.step`]. If `None`, nothing is passed
                downstream to the scheduler (use `None` for schedulers which don't support this argument).
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            seg_batch (`torch.Tensor`, *optional*):
                batch of segmentations to condition generation on
            class_label_cfg (`int`, *optional*, defaults to `None`):
                class label to condition generation on using CFG, if using class-conditional model

            OPTIONS FOR IMAGE TRANSLATION:
            translate (`bool`, *optional*, defaults to False):
                whether to translate images from the source domain to the target domain

        Example:

        ```py

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        # Sample gaussian noise to begin loop
        if self.external_config.config['segmentation_channel_mode'] == "single":
            img_channel_ct = self.unet.config.in_channels - 1
        elif self.external_config.config['segmentation_channel_mode'] == "multi":
            img_channel_ct = self.unet.config.in_channels - len([k for k in seg_batch.keys() if k.startswith("seg_")])

        if self.external_config.config['neighboring_images_guided']:    # discard clean_left and clean_right for denoising scheduler
            img_channel_ct -= 2
        
        if self.external_config.config['adc_guided']:
            img_channel_ct -= 1 

        if isinstance(self.unet.config.sample_size, int):
            if self.external_config.config['segmentation_channel_mode'] == "single":
                if self.external_config.config['neighboring_images_guided']:        # -1 for seg, -2 for neighboring slices
                    image_shape = (
                        batch_size,
                        img_channel_ct,
                        self.unet.config.sample_size,
                        self.unet.config.sample_size,
                    )
                else:
                    image_shape = (
                        batch_size,
                        img_channel_ct,
                        self.unet.config.sample_size,
                        self.unet.config.sample_size,
                    )
            elif self.external_config.config['segmentation_channel_mode'] == "multi":
                image_shape = (
                    batch_size,
                    self.unet.config.in_channels - len([k for k in seg_batch.keys() if k.startswith("seg_")]),
                    self.unet.config.sample_size,
                    self.unet.config.sample_size,
                )
        else:
            if self.external_config.config['segmentation_channel_mode'] == "single":
                if self.external_config.config['neighboring_images_guided']:        # -1 for seg, -2 for neighboring slices
                    image_shape = (
                        batch_size,
                        img_channel_ct,
                        *self.unet.config.sample_size,
                    )
                else:
                    image_shape = (
                        batch_size,
                        img_channel_ct
                        *self.unet.config.sample_size,
                    )

            elif self.external_config.config['segmentation_channel_mode'] == "multi":
                image_shape = (batch_size, self.unet.config.in_channels - len([k for k in seg_batch.keys() if k.startswith("seg_")]), *self.unet.config.sample_size)
            
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # initiate latent variable to sample from
        if not translate:
            # normal sampling; start from noise
            image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)
        else:
            # image translation sampling; start from source domain images, add noise up to certain step, then being there for denoising
            trans_start_t = int(self.external_config.config['trans_noise_level'] * self.scheduler.config.num_train_timesteps)

            trans_start_images = seg_batch["images"].to(self._execution_device)

            # Sample noise to add to the images
            noise = torch.randn(trans_start_images.shape).to(trans_start_images.device)
            timesteps = torch.full(
                (trans_start_images.size(0),),
                trans_start_t, 
                device=trans_start_images.device
                ).long()
            image = self.scheduler.add_noise(trans_start_images, noise, timesteps)

        if self.external_config['use_repaint']:
            ts = get_schedule_jump_paper()
        else:
            self.scheduler.set_timesteps(num_inference_steps)
            ts = list(self.scheduler.timesteps.cpu().numpy())

        for t in self.progress_bar(self.scheduler.timesteps):
            if translate:
                # if doing translation, start at chosen time step given partially-noised image
                # skip all earlier time steps (with higher t)
                if t >= trans_start_t:
                  continue

            # 1. predict noise model_output
            # first, concat segmentations to noise
            image = add_segmentations_to_noise(image, seg_batch, self.external_config.config, self.device)

            # 2. Concatenate neighboring slices
            if self.external_config.config['neighboring_images_guided']:
                image = add_neighboring_images_to_noise(image, seg_batch, self.external_config.config, self.device)

            # 3. Concatenate adc slice
            if self.external_config.config['adc_guided']:
                image = add_adc_to_noise(image, seg_batch, self.external_config.config, self.device)

            if self.external_config.config['class_conditional']:
                true_class_labels = seg_batch['class_label'].long().to(self.device)
                
                # use cfg for eval
                if self.external_config.config['cfg_eval']:

                    # use fake_class_labels for synthesis
                    if fake_class_labels is not None:       
                        fake_class_labels = fake_class_labels.long().to(self.device)
                        model_output_cond = self.unet(image, t, class_labels=fake_class_labels).sample
                        model_output_uncond = self.unet(image, t, class_labels=torch.zeros_like(fake_class_labels).long()).sample
                    
                    # use true_class_labels for cfg
                    else:       
                        model_output_cond = self.unet(image, t, class_labels=true_class_labels).sample
                        model_output_uncond = self.unet(image, t, class_labels=torch.zeros_like(true_class_labels).long()).sample

                    model_output = (1. + self.external_config.config['cfg_weight']) * model_output_cond - self.external_config.config['cfg_weight'] * model_output_uncond
                
                else:
                    # no cfg for eval
                    # TODO: complete no use cfg and synthesize using fake labels
                    model_output = self.unet(image, t, class_labels=true_class_labels).sample
            else:
                # no class guided
                model_output = self.unet(image, t).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            # but first, we're only adding denoising the image channel (not seg channel),
            # so remove segs
            image = image[:, :img_channel_ct, :, :]
            step_output = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            )
            x_t_minus_one = step_output.prev_sample     # this is \hat{x}_{t-1}

            if self.external_config.config['use_repaint']:
                original_images = seg_batch['image']
                noise = torch.randn(original_images.shape).to(original_images.device)

                t_minus_one = torch.full((original_images.shape[0],), t - 1, device=self.device, dtype=torch.long)

                x_tminus1_known = self.scheduler.add_noise(original_images, noise, t_minus_one)

                for seg_type in seg_batch.keys():
                    if seg_type.startswith("seg_"):     # Careful, in_paint mask being the 1st element starts with "seg_"
                        inpaint_mask = (seg_batch[seg_type] > 0).float()       # shape = [B, H, W]
                        print("line 1031", inpaint_mask.shape)

            image = x_t_minus_one

        # image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


class SegGuidedDDPMRepaintPipeline(DiffusionPipeline):
    unet: UNet2DModel
    scheduler: RePaintScheduler

    def __init__(self, unet, scheduler, eval_dataloader, external_config):
        super().__init__()
        # Wrap the external_config so it behaves like a module
        # wrapped_external_config = SimpleNamespace(**external_config)
        wrapped_external_config = ConfigWrapper(external_config)
        self.register_modules(unet=unet, scheduler=scheduler, eval_dataloader=eval_dataloader, external_config=wrapped_external_config)
        # ^ some reason necessary for DDIM but not DDPM.

        self.eval_dataloader = eval_dataloader
        self.external_config = wrapped_external_config # config is already a thing

        # make sure scheduler can always be converted to DDIM
        # scheduler = DDIMScheduler.from_config(scheduler.config)

    # def __init__(self, unet, scheduler):
    #     super().__init__()
    #     self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        # original_image: Union[torch.FloatTensor, PIL.Image.Image],
        # mask_image: Union[torch.FloatTensor, PIL.Image.Image],
        num_inference_steps: int = 250,
        eta: float = 0.0,
        jump_length: int = 10,
        jump_n_sample: int = 10,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "numpy",
        return_dict: bool = True,
        seg_batch: Optional[torch.Tensor] = None,
        # class_label_cfg: Optional[int] = None,
        fake_class_labels: Optional[torch.Tensor] = None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            original_image (`torch.FloatTensor` or `PIL.Image.Image`):
                The original image to inpaint on.
            mask_image (`torch.FloatTensor` or `PIL.Image.Image`):
                The mask_image where 0.0 values define which part of the original image to inpaint (change).
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            eta (`float`):
                The weight of noise for added noise in a diffusion step. Its value is between 0.0 and 1.0 - 0.0 is DDIM
                and 1.0 is DDPM scheduler respectively.
            jump_length (`int`, *optional*, defaults to 10):
                The number of steps taken forward in time before going backward in time for a single jump ("j" in
                RePaint paper). Take a look at Figure 9 and 10 in https://arxiv.org/pdf/2201.09865.pdf.
            jump_n_sample (`int`, *optional*, defaults to 10):
                The number of times we will make forward time jump for a given chosen time sample. Take a look at
                Figure 9 and 10 in https://arxiv.org/pdf/2201.09865.pdf.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """
       
        # Sample gaussian noise to begin loop
        if self.external_config.config['segmentation_channel_mode'] == "single":
            img_channel_ct = self.unet.config.in_channels - 1
        elif self.external_config.config['segmentation_channel_mode'] == "multi":
            img_channel_ct = self.unet.config.in_channels - len([k for k in seg_batch.keys() if k.startswith("seg_")])

        if self.external_config.config['neighboring_images_guided']:    # discard clean_left and clean_right for denoising scheduler
            img_channel_ct -= 2
        
        if self.external_config.config['adc_guided']:
            img_channel_ct -= 1 

        if isinstance(self.unet.config.sample_size, int):
            if self.external_config.config['segmentation_channel_mode'] == "single":
                if self.external_config.config['neighboring_images_guided']:        # -1 for seg, -2 for neighboring slices
                    image_shape = (
                        batch_size,
                        img_channel_ct,
                        self.unet.config.sample_size,
                        self.unet.config.sample_size,
                    )
                else:
                    image_shape = (
                        batch_size,
                        img_channel_ct,
                        self.unet.config.sample_size,
                        self.unet.config.sample_size,
                    )
            elif self.external_config.config['segmentation_channel_mode'] == "multi":
                image_shape = (
                    batch_size,
                    self.unet.config.in_channels - len([k for k in seg_batch.keys() if k.startswith("seg_")]),
                    self.unet.config.sample_size,
                    self.unet.config.sample_size,
                )
        else:
            if self.external_config.config['segmentation_channel_mode'] == "single":
                if self.external_config.config['neighboring_images_guided']:        # -1 for seg, -2 for neighboring slices
                    image_shape = (
                        batch_size,
                        img_channel_ct,
                        *self.unet.config.sample_size,
                    )
                else:
                    image_shape = (
                        batch_size,
                        img_channel_ct
                        *self.unet.config.sample_size,
                    )

            elif self.external_config.config['segmentation_channel_mode'] == "multi":
                image_shape = (batch_size, self.unet.config.in_channels - len([k for k in seg_batch.keys() if k.startswith("seg_")]), *self.unet.config.sample_size)
            
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # initiate latent variable to sample from
        # normal sampling; start from noise
        image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps, jump_length, jump_n_sample, self.device)
        self.scheduler.eta = eta

        t_last = self.scheduler.timesteps[0] + 1
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            
            # 1. predict noise model_output
            # first, concat segmentations to noise
            image = add_segmentations_to_noise(image, seg_batch, self.external_config.config, self.device)

            # 2. Concatenate neighboring slices
            if self.external_config.config['neighboring_images_guided']:
                image = add_neighboring_images_to_noise(image, seg_batch, self.external_config.config, self.device)

            # 3. Concatenate adc slice
            if self.external_config.config['adc_guided']:
                image = add_adc_to_noise(image, seg_batch, self.external_config.config, self.device)
            
            if t < t_last:
                    
                if self.external_config.config['class_conditional']:
                    true_class_labels = seg_batch['class_label'].long().to(self.device)
        
                    # use cfg for eval
                    if self.external_config.config['cfg_eval']:
                        # use fake_class_labels for synthesis
                        if fake_class_labels is not None:       
                            fake_class_labels = fake_class_labels.long().to(self.device)
                            model_output_cond = self.unet(image, t, class_labels=fake_class_labels).sample
                            model_output_uncond = self.unet(image, t, class_labels=torch.zeros_like(fake_class_labels).long()).sample

                        else:       # use true_class_labels for cfg  
                            model_output_cond = self.unet(image, t, class_labels=true_class_labels).sample
                            model_output_uncond = self.unet(image, t, class_labels=torch.zeros_like(true_class_labels).long()).sample

                        model_output = (1. + self.external_config.config['cfg_weight']) * model_output_cond - self.external_config.config['cfg_weight'] * model_output_uncond
                    
                    else:
                        # no cfg for eval
                        # TODO: complete no use cfg and synthesize using fake labels
                        model_output = self.unet(image, t, class_labels=true_class_labels).sample
                else:
                    # no class guided
                    model_output = self.unet(image, t).sample

                
                x_in = image[:, :img_channel_ct, :, :]
                step_out = self.scheduler.step(
                    model_output, t, x_in,
                    eta=eta,
                    use_clipped_model_output=use_clipped_model_output,
                    generator=generator
                )
                image = step_out.prev_sample
            
            else:
                t_last = t

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)