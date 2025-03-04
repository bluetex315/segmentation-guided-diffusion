from PIL import Image
import os
import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def load_config(config_file):
    """Load a YAML configuration file and return a nested dictionary."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def flatten_config(config):
    """
    Merge nested configuration dictionaries into a single flat dictionary.
    
    For each top-level section (e.g., model_args, data_args, etc.), the inner
    dictionary is merged into a single dictionary. If there are duplicate keys,
    later sections will overwrite the earlier ones.
    """
    flat_config = {}
    for section, sub_config in config.items():
        if isinstance(sub_config, dict):
            flat_config.update(sub_config)
        else:
            flat_config[section] = sub_config
    return flat_config


def parse_3d_volumes(dset_dict, seg_type, csv_file=None):

    if csv_file is not None:
        labels_df = pd.read_csv(csv_file)
        # Expecting columns: 'patient_id', 'slice_idx', 'class_label'
        labels_dict = {}
        for _, row in labels_df.iterrows():
            # Convert patient_id to string and slice_idx to integer for consistency.
            key = (f"{int(row['fastmri_pt_id']):03d}", int(row['slice']-1))     # row['slice']-1 to match the Python indexing
            labels_dict[key] = row['PIRADS']
    else:
        labels_dict = {}
    
    patient_dict = {}
    
    # Process the original image paths.
    image_key = 'image'
    if image_key in dset_dict:
        for img_path in dset_dict[image_key]:
            # Extract the patient ID.
            basename = os.path.basename(img_path)  # e.g., "110_T2W.nii.gz"
            patient_id = basename.split('_')[0]    # e.g., "110"
            # Create an entry for the patient if not already present.
            if patient_id not in patient_dict:
                patient_dict[patient_id] = {}
            patient_dict[patient_id][image_key] = img_path
            
    # Process segmentation paths (any key that is not "image").
    for key in dset_dict:
        if key == image_key:
            continue
        for seg_path in dset_dict[key]:
            # Extract the patient ID.
            basename = os.path.basename(seg_path)  # e.g., "110_T2W_infer_seg.nii.gz"
            patient_id = basename.split('_')[0]     # e.g., "110"
            if patient_id not in patient_dict:
                patient_dict[patient_id] = {}
            patient_dict[patient_id][key] = seg_path

    selected_slices = []
    
    for patient_id, files in patient_dict.items():
        # Check that both image and segmentation exist for this patient.

        seg_key = 'seg_'+seg_type
        # Load the 3D volumes.
        img_vol = nib.load(files['image']).get_fdata()
        seg_vol = nib.load(files[seg_key]).get_fdata()
    
        # Optionally check that image and segmentation have the same number of slices.
        if img_vol.shape[2] != seg_vol.shape[2]:
            print(f"Warning: patient {patient_id} image and segmentation volumes have different number of slices.")
            continue
        
        num_slices = img_vol.shape[2]
        
        # Identify slice indices where the segmentation is "active"
        valid_indices = [z for z in range(seg_vol.shape[2]) if np.count_nonzero(seg_vol[:, :, z]) > 0]
        
        if not valid_indices:
            print(f"No valid segmentation found for patient {patient_id}. Skipping patient.")
            continue
        
        # Determine the selection range.
        # For example, if the active segmentation is between slice 8 and 22, then include from
        # max(0, 8-neighbor_range) to min(num_slices-1, 22+neighbor_range)
        neighbor_range = 3
        min_valid = min(valid_indices)
        max_valid = max(valid_indices)
        start_idx = max(0, min_valid - neighbor_range)
        end_idx = min(num_slices - 1, max_valid + neighbor_range)
        
        # For each slice in the computed range, save the image and segmentation slices.
        for z in range(start_idx, end_idx + 1):
            slice_info = {
                'patient_id': patient_id,
                'slice_idx': z,
                'image': img_vol[:, :, z],
                seg_key: seg_vol[:, :, z]
            }

            # Add the slice-level class label if available.
            key = (patient_id, z)

            if key in labels_dict:
                slice_info['class_label'] = labels_dict[key]
            else:
                print("None", key)
                slice_info['class_label'] = None  # or you might choose to skip the slice if no label exists.
            # print(slice_info)
            selected_slices.append(slice_info)
    
    return selected_slices



# Split the dataset dictionary into train and test splits
def train_test_split_dset(dset_dict, test_size=0.2, random_state=42):
    # Create indices for splitting
    if '/home/lc2382/project/fastMRI_NYU/nifti/223/223_T2W.nii.gz' in dset_dict['image']:
        dset_dict['image'].remove('/home/lc2382/project/fastMRI_NYU/nifti/223/223_T2W.nii.gz')
        print(f'Removing problematic image')
    num_samples = len(dset_dict[list(dset_dict.keys())[0]])
    indices = list(range(num_samples))
    
    # Perform the split
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)

    # Create train and test dictionaries by indexing into dset_dict
    train_dict = {key: [dset_dict[key][i] for i in train_indices] for key in dset_dict}
    test_dict = {key: [dset_dict[key][i] for i in test_indices] for key in dset_dict}

    # print(train_dict)

    return train_dict, test_dict


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def save_nifti(batch, seg_key, output_dir):
    """
    Save a batch of 2D slices as individual NIfTI files.

    Parameters
    ----------
    batch : torch.Tensor
        A tensor of shape [batch_size, channels, height, width], containing the image slices.
    seg_key: str
        Key to index segmentation mask from batch
    output_dir : str, optional
        The directory where the NIfTI files will be saved. Defaults to the current directory.

    Notes
    -----
    - This function assumes that each slice in `batch` corresponds to one patient ID and one slice index.
    - The `affine` used here is a simple identity matrix. Modify it if you have a known orientation and spacing.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    batch_size = len(batch['patient_id'])

    seg_guided = True if seg_key else False # Check whether we have segmentation data

    for i in range(batch_size):
        pid = batch['patient_id'][i]
        sidx = batch['slice_idx'][i]

        # Extract the i-th slice (assuming a single channel for image)
        img_tensor = batch['image'][i, 0, ...]  # shape: [height, width]
        img_np = img_tensor.cpu().numpy()  # Convert to NumPy

        # Create a NIfTI image for the main image
        nifti_img = nib.Nifti1Image(img_np, affine=np.eye(4))

        # Construct the output filename for the main image
        image_output_filename = os.path.join(output_dir, f"{pid}_slice_{sidx}_T2W.nii.gz")
        nib.save(nifti_img, image_output_filename)
        print(f"Saved Image: {image_output_filename}")

        # Save segmentation only if it exists
        if seg_guided:
            seg_tensor = batch[seg_key][i, 0, ...]
            seg_np = seg_tensor.cpu().numpy()
            seg_img = nib.Nifti1Image(seg_np, affine=np.eye(4))
            seg_output_filename = os.path.join(output_dir, f"{pid}_slice_{sidx}_{seg_key}.nii.gz")
            nib.save(seg_img, seg_output_filename)
            print(f"Saved Segm: {seg_output_filename}")