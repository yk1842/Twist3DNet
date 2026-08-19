import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import Dataset, Sampler


class MyNiiDataset(Dataset):
    def __init__(self, root_dir, transform=None, step='train',
                 target_shape=(1, 13, 256, 256), nifti_axis_order='HWD'):
        self.root_dir = root_dir
        self.transform = transform
        self.step = step
        self.target_shape = target_shape
        self.nifti_axis_order = nifti_axis_order.upper()

        classes_folder = os.path.join(self.root_dir, self.step)
        if not os.path.isdir(classes_folder):
            raise FileNotFoundError(f'Dataset directory not found: {classes_folder}')

        # Sort class names to ensure reproducible label mapping
        classes = sorted([
            x for x in os.listdir(classes_folder)
            if os.path.isdir(os.path.join(classes_folder, x))
        ])

        if len(classes) != 2:
            raise ValueError(f'HPC classification requires 2 classes, but found: {classes}')

        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        print(f'MyNiiDataset: class_to_idx = {self.class_to_idx}')

        # Collect .nii and .nii.gz files
        self.images = sorted(
            glob.glob(os.path.join(classes_folder, '*', '*.nii')) +
            glob.glob(os.path.join(classes_folder, '*', '*.nii.gz'))
        )

        if len(self.images) == 0:
            raise RuntimeError(f'No NIfTI files found in {classes_folder}')

        self.labels = {}
        self.targets = []

        for path in self.images:
            class_name = os.path.basename(os.path.dirname(path))
            label = self.class_to_idx[class_name]
            self.labels[path] = label
            self.targets.append(label)

        print(f'Loaded {len(self.images)} samples.')

    def load_image(self, file_path):
        image = nib.load(file_path).get_fdata().astype(np.float32)

        if image.ndim != 3:
            raise ValueError(f'Expected 3D NIfTI image, but got {image.shape}: {file_path}')

        # nibabel commonly returns (H, W, D), convert it to (D, H, W)
        if self.nifti_axis_order == 'HWD':
            image = np.transpose(image, (2, 0, 1))
        elif self.nifti_axis_order != 'DHW':
            raise ValueError("nifti_axis_order must be 'HWD' or 'DHW'")

        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).float().unsqueeze(0)  # (C, D, H, W)
        return image

    def resize_depth(self, image, target_depth):
        _, depth, height, width = image.shape

        # D > 13: volumetric reduction using trilinear interpolation
        if depth > target_depth:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(target_depth, height, width),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)

        # D < 13: zero padding to target depth
        elif depth < target_depth:
            pad_total = target_depth - depth
            pad_front = pad_total // 2
            pad_back = pad_total - pad_front
            image = F.pad(image, (0, 0, 0, 0, pad_front, pad_back),
                          mode='constant', value=0)

        return image

    def resize_inplane(self, image, target_height, target_width):
        _, depth, height, width = image.shape

        # Preserve original aspect ratio
        scale = min(target_height / height, target_width / width)
        new_height = max(1, int(round(height * scale)))
        new_width = max(1, int(round(width * scale)))

        if new_height != height or new_width != width:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(depth, new_height, new_width),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)

        # Zero padding to 256 x 256
        pad_h = target_height - new_height
        pad_w = target_width - new_width
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        image = F.pad(
            image,
            (pad_left, pad_right, pad_top, pad_bottom, 0, 0),
            mode='constant',
            value=0
        )

        return image

    def apply_transform(self, image):
        if self.transform is None:
            return image

        # Our format: (C, D, H, W)
        # TorchIO format: (C, W, H, D)
        image_tio = image.permute(0, 3, 2, 1).contiguous()

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image_tio)
        )

        subject = self.transform(subject)
        image_tio = subject['image'].data

        # Convert back to (C, D, H, W)
        image = image_tio.permute(0, 3, 2, 1).contiguous()
        return image

    def zscore_normalization(self, image, eps=1e-8):
        # Normalize non-zero voxels so that padding remains zero
        mask = image != 0

        if mask.any():
            values = image[mask]
            mean = values.mean()
            std = values.std(unbiased=False)

            image = image.clone()

            if std > eps:
                image[mask] = (values - mean) / (std + eps)
            else:
                image[mask] = values - mean

        return image

    def __getitem__(self, index):
        file_path = self.images[index]
        label = self.labels[file_path]

        image = self.load_image(file_path)

        _, target_depth, target_height, target_width = self.target_shape

        # Standardize RoI to 13 x 256 x 256
        image = self.resize_depth(image, target_depth)
        image = self.resize_inplane(image, target_height, target_width)

        # Online augmentation is applied only when transform is provided
        image = self.apply_transform(image)

        # Z-score normalization
        image = self.zscore_normalization(image)

        if tuple(image.shape) != tuple(self.target_shape):
            raise RuntimeError(
                f'Unexpected image shape {tuple(image.shape)}, '
                f'expected {self.target_shape}: {file_path}'
            )

        return image.float(), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.images)


class ResampleSubsetSampler(Sampler):
    """
    Repeat the training subset multiple times per epoch.
    For HPC, num_resamples=15 according to the experimental setting.
    """
    def __init__(self, indices, num_resamples=15, seed=42):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.num_resamples = num_resamples
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.num_resamples):
            indices = self.rng.permutation(self.indices)
            yield from indices.tolist()

    def __len__(self):
        return len(self.indices) * self.num_resamples
