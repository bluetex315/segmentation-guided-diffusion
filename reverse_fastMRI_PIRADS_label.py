import pandas as pd

# -------------------------------------------------------------------
# Paths (edit these as needed)
input_csv  = "/home/lc2382/project/segmentation-guided-diffusion/data/labels/t2_slice_level_labels.csv"
output_csv = "/home/lc2382/project/segmentation-guided-diffusion/data/labels/t2_slice_level_labels_reversed.csv"
# -------------------------------------------------------------------

# 1) Load the original slice‑level CSV
df = pd.read_csv(input_csv)

# 2) Compute the maximum slice index for each patient
#    (assumes column 'fastmri_pt_id' for patient and 'slice' for slice number)
max_slice = df.groupby('fastmri_pt_id')['slice'].transform('max')

# 3) Reverse each slice index: new_slice = max_slice - old_slice + 1
df['slice'] = max_slice - df['slice'] + 1

# 4) Save out the corrected CSV
df.to_csv(output_csv, index=False)

print(f"Reversed‑slice CSV written to: {output_csv}")
