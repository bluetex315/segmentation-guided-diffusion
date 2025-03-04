from PIL import Image
import os
import nibabel as nib
import numpy as np

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