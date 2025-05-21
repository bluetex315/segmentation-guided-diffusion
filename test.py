from PIL import Image
import os
import re
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, NormalizeIntensityd, ToTensord, Orientationd, CenterSpatialCropd, Orientationd, ScaleIntensityRangePercentilesd


class NiftiSliceDataset(Dataset):
    def __init__(self, root_dir, mode='orig', transform=None):
        """
        Args:
            root_dir (str): Root path containing sample subdirs.
            mode (str): 'orig' to load original (_orig.nii.gz),
                        'fake' to load synthetic (_syn.nii.gz).
            transform (callable, optional): transforms to apply.
        """
        self.paths = []
        pattern = r'_orig\.nii\.gz$' if mode == 'orig' else r'_syn\.nii\.gz$'
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if re.search(pattern, fname):
                    self.paths.append(os.path.join(dirpath, fname))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        # print("line42", path)
        img = nib.load(path).get_fdata().astype(np.float32)
        img = np.squeeze(img)
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        # parse class label from filename
        m = re.search(r'class(\d+)', os.path.basename(path))
        label = int(m.group(1)) if m else -1
        sample = {'image': img, 'label': label}
        # print("line 50", sample)
        return self.transform(sample) if self.transform else sample

def preprocess_for_inception(batch):
    """
    Args:
        batch: dict with 'image': Tensor (B, C, H, W)
    Returns:
        Tensor (B,3,299,299) normalized for Inception.
    """
    imgs = batch['image']
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)
    # ensure 3 channels
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
    # resize to 299×299
    imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    # scale [–1,1] or [0,1] → [0,1]
    if imgs.min() < 0 or imgs.max() > 1:
        imgs = (imgs + 1) / 2
    # normalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device)[None,:,None,None]
    std  = torch.tensor([0.229, 0.224, 0.225], device=imgs.device)[None,:,None,None]
    imgs = (imgs - mean) / std
    return imgs

class FIDLoader:
    def __init__(self, base_loader):
        self.loader = base_loader

    def __iter__(self):
        for batch in self.loader:
            yield preprocess_for_inception(batch)

@torch.no_grad()
def get_activations(loader, model, device):
    model.eval()
    acts = []
    for imgs in tqdm(loader, desc="Extracting FID features"):
        imgs = imgs.to(device)
        feat = model(imgs)  # (B, 2048)
        acts.append(feat.cpu().numpy())
    return np.concatenate(acts, axis=0)

def compute_fid(real_loader, fake_loader, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    # load InceptionV3
    inc = inception_v3(pretrained=True, transform_input=False).to(device)
    inc.fc = nn.Identity()
    inc.eval()
    
    # extract features
    real_feats = get_activations(real_loader, inc, device)
    fake_feats = get_activations(fake_loader, inc, device)
    
    # compute stats
    mu_r, sigma_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
    
    # mean difference
    diff = mu_r - mu_f
    diff_sq = diff.dot(diff)
    # sqrt of covariance product
    covmean, _ = sqrtm(sigma_r.dot(sigma_f), disp=False)
    covmean = covmean.real if np.iscomplexobj(covmean) else covmean
    fid = diff_sq + np.trace(sigma_r + sigma_f - 2 * covmean)
    
    return float(fid)

def compute_fid_per_class(root_dir, transform, cls, **fid_kwargs):
    """
    Compute FID only on samples whose filename contains 'class{cls}'.
    """
    # 1) build the full datasets
    real_ds = NiftiSliceDataset(root_dir, mode='orig', transform=transform)
    fake_ds = NiftiSliceDataset(root_dir, mode='fake', transform=transform)

    # 2) pick only the indices matching your class
    real_idx = [i for i,p in enumerate(real_ds.paths) if f"class{cls}_" in p]
    fake_idx = [i for i,p in enumerate(fake_ds.paths) if f"class{cls}_" in p]
    print("line 129", real_idx)
    print("line 130", fake_idx)

    # 3) split into two roughly equal halves
    mid = len(real_idx) // 2
    idx1, idx2 = real_idx[:mid], real_idx[mid:]

    # 4) wrap them in Subsets
    real_sub1 = Subset(real_ds, idx1)
    real_sub2 = Subset(real_ds, idx2)

    # 5) build loaders
    loader1 = DataLoader(real_sub1, batch_size=32, shuffle=False, num_workers=4)
    loader2 = DataLoader(real_sub2, batch_size=32, shuffle=False, num_workers=4)

    # 6) wrap for Inception and compute
    fid1 = FIDLoader(loader1)
    fid2 = FIDLoader(loader2)

    return compute_fid(fid1, fid2, **fid_kwargs)

    # # 3) wrap them in Subsets
    # real_sub = Subset(real_ds, real_idx)
    # fake_sub = Subset(fake_ds, fake_idx)

    # # 4) build loaders
    # real_loader = DataLoader(real_sub, batch_size=32, shuffle=False, num_workers=4)
    # fake_loader = DataLoader(fake_sub, batch_size=32, shuffle=False, num_workers=4)

    # # 5) wrap for Inception and compute
    # real_fid = FIDLoader(real_loader)
    # fake_fid = FIDLoader(fake_loader)
    
    # return compute_fid(real_fid, fake_fid, **fid_kwargs)


root = "/home/lc2382/project/segmentation-guided-diffusion/experiments/ddim-fastMRI_NYU-128-segguided-classCond-CFG_20250507_234022/samples/0399"

# Example: FID for PI‑RADS class X only
fid_cls = compute_fid_per_class(
    root,
    None,
    cls=4,
    device='cuda'
)
print(f"Class FID: {fid_cls:.2f}")
