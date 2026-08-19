# Twist3DNet
Twist3DNet: a 2D-3D hybrid network based on transfer learning for prognosis classification of hypopharyngeal cancer. This config is used on HPC dataset(based on mmpretrain) And the checkpoint file of this config is normal pretrained weights of ResNet18-3D, just load it.

We have updated the repository with the latest implementation of Twist3DNet for hypopharyngeal cancer prognosis classification. In addition to the original 2D–3D hybrid framework based on MMPreTrain, this update provides the simple model code for Twist3DNet + ResNet18-2D, as well as additional 2D branch implementations including ShuffleNetV2-2D, SENet18-2D, and ConvMixer-2D. Furthermore, two modified hybrid networks, H-DenseUNet for classification and MDU-Net for classification, are included as comparative models to facilitate fair performance evaluation against the proposed Twist3DNet framework.



## Table of Contents

- [Environment Dependencies](#environment-dependencies)
- [Data Preparation](#data-preparation)
  - [HPC Dataset (Private)](#hpc-dataset-private)
  - [BraTS2018 Dataset](#brats2018-dataset)
  - [3DLSC-COVID Dataset](#3dlsc-covid-dataset)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)


## Environment Dependencies

This project requires the following dependencies. We recommend using a virtual environment (e.g., `conda` or `venv`) for installation.

### Core Dependencies

| Package | Version | Package | Version |
|:---|:---|:---|:---|
| absl-py | 2.4.0 | kiwisolver | 1.5.0 |
| astunparse | 1.6.3 | libclang | 18.1.1 |
| certifi | 2026.4.22 | markdown | 3.10.2 |
| charset-normalizer | 3.4.7 | markdown-it-py | 4.2.0 |
| click | 8.3.3 | markupsafe | 3.0.3 |
| contourpy | 1.3.3 | mat4py | 0.6.0 |
| cycler | 0.12.1 | matplotlib | 3.10.9 |
| deprecated | 1.3.1 | ml-dtypes | 0.5.4 |
| einops | 0.8.2 | mmpretrain | 1.2.0 |
| filelock | 3.29.0 | networkx | 3.6.1 |
| flatbuffers | 25.12.19 | nibabel | 5.4.2 |
| fonttools | 4.62.1 | numba | 0.67.0 |
| fsspec | 2026.4.0 | numpy | 2.4.4 |
| gast | 0.7.0 | opencv-python | 4.13.0.92 |
| google-pasta | 0.2.0 | openpyxl | 3.1.5 |
| grpcio | 1.80.0 | opt-einsum | 3.4.0 |
| h5py | 3.14.0 | optree | 0.19.1 |
| humanize | 4.15.0 | packaging | 26.0 |
| idna | 3.14 | pandas | 3.0.2 |
| importlib-metadata | 9.0.0 | pillow | 12.2.0 |
| jaxtyping | 0.3.9 | pip | 26.0.1 |
| jinja2 | 3.1.6 | protobuf | 7.34.1 |
| joblib | 1.5.3 | pygments | 2.20.0 |
| keras | 3.14.1 | pynndescent | 0.6.0 |
| | | pyparsing | 3.3.2 |

| Package | Version | Package | Version |
|:---|:---|:---|:---|
| python | 3.12.13 | thop | 0.1.1 |
| python-dateutil | 2.9.0.post0 | threadpoolctl | 3.6.0 |
| pyyaml | 6.0.3 | torch | 2.5.1+cu121 |
| requests | 2.33.1 | torchaudio | 2.11.0 |
| rich | 15.0.0 | torchio | 1.2.0 |
| scikit-learn | 1.8.0 | torchvision | 0.20.1+cu121 |
| scipy | 1.17.1 | tqdm | 4.67.3 |
| setuptools | 81.0.0 | triton | 3.1.0 |
| shellingham | 1.5.4 | typer | 0.25.1 |
| simpleitk | 2.5.4 | typing-extensions | 4.15.0 |
| six | 1.17.0 | umap-learn | 0.5.12 |
| sympy | 1.13.1 | urllib3 | 2.7.0 |
| tensorflow | 2.21.0 | wheel | 0.46.3 |
| termcolor | 3.3.0 | wrapt | 2.1.2 |

> **Note:** System-level dependencies (e.g., `bzip2`, `libgcc`, `openssl`, `zlib`, etc.) are omitted from the list above. They will be automatically handled by the package manager during environment setup.

### Installation

```bash
# Clone the repository
git clone https://github.com/yk1842/Twist3DNet.git
cd Twist3DNet

# Create and activate conda environment
conda create -n twistnet python=3.12
conda activate twistnet

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

This project uses three medical imaging datasets: a private hypopharyngeal cancer (HPC) MRI dataset and two publicly available datasets, BraTS2018 and 3DLSC-COVID. The datasets were prepared as described below.

### HPC Dataset (Private)

The **HPC dataset** is a private clinical dataset collected from patients with pathologically confirmed hypopharyngeal squamous cell carcinoma treated at the Cancer Hospital of Chinese Academy of Medical Sciences between 2010 and 2015. It contains T1- and T2-weighted MRI scans together with clinical follow-up information.

After applying the inclusion and exclusion criteria and removing cases with insufficient prognostic information, **119 patients** were included for prognosis classification. According to the survival threshold used in this study, the patients were divided into:

* **Short-survival group (≤735 days):** 46 patients
* **Long-survival group (>735 days):** 73 patients

Five-fold cross-validation was used for model evaluation. The dataset was randomly divided into five approximately equal folds at the patient level, with four folds used for training and one fold used for validation in each round.

For preprocessing, each Region of Interest (RoI) was standardized to **13×256×256 (D×H×W)**. RoIs with more than 13 slices were resized using trilinear interpolation, while RoIs with fewer than 13 slices were padded to the required depth. The in-plane dimensions were proportionally resized and zero-padded to 256×256 when necessary. Z-score normalization was applied to image intensities.

During training, online data augmentation, including random translation, rotation, and flipping, was applied. The training dataset underwent **15 rounds of resampling**, and the batch size was set to **16**.

> **Data Availability:** Due to patient privacy and institutional ethical restrictions, the HPC dataset cannot be publicly released.

### BraTS2018 Dataset

The **BraTS2018 dataset** is a publicly available multimodal brain tumor MRI dataset provided by the MICCAI BraTS 2018 challenge. Among the available cases, **285 MRI scans with expert tumor annotations** were used in this study.

The **Whole Tumor (WT)** region was selected as the tumor mask. Slices without WT regions were removed, and the RoI of each MRI modality was extracted according to the corresponding mask.

The dataset was divided into:

* **Training set:** 231 patients

  * 171 High-Grade Glioma (HGG)
  * 60 Low-Grade Glioma (LGG)
* **Validation set:** 54 patients

  * 39 HGG
  * 15 LGG

All volumes were resized to **152×244×244 (D×H×W)** for model input. The batch size was set to **8**.

The original dataset can be obtained from **MICCAI BraTS 2018: Data**(https://www.med.upenn.edu/sbia/brats2018/). Please follow the official BraTS instructions to obtain the dataset before running the preprocessing and training procedures.

### 3DLSC-COVID Dataset

The **3DLSC-COVID dataset** is a publicly available 3D chest CT dataset containing COVID-19, Community-Acquired Pneumonia (CAP), and healthy cases. The complete dataset contains 1,805 CT volumes, including 794 COVID-19 cases, 540 CAP cases, and 471 healthy individuals.

Since only part of the dataset is publicly released, this study used the publicly available subset containing **100 COVID-19 CT scans and 32 CAP CT scans**.

The dataset was divided into:

* **Training set:** 89 cases

  * 67 COVID-19
  * 22 CAP
* **Validation set:** 43 cases

  * 33 COVID-19
  * 10 CAP

During training, the training dataset underwent **two rounds of resampling**, and the batch size was set to **16**.

The publicly available data can be obtained from the **DeepSC-COVID** GitHub repository(https://github.com/XiaofeiWang2018/DeepSC-COVID). Please follow the instructions provided in the original repository to download the released 3DLSC-COVID data.

## Usage

### Model Implementation

The main implementation of Twist3DNet is provided in:

`twist3dnet with resnet.py`

The default 2D branch is ResNet18-2D. Alternative 2D branches, including ShuffleNetV2-2D, SENet18-2D, and ConvMixer-2D, are provided in:

`2d branches.py`

Users can replace the default 2D branch with the corresponding implementation according to their experimental requirements.

### Data Preparation

The repository does not include the original clinical data or dataset preprocessing scripts. Please prepare the HPC, BraTS2018, and 3DLSC-COVID datasets according to the procedures described in the [Data Preparation](#data-preparation) section.

### Pre-trained Weights

ImageNet-pretrained weights for standard 2D backbones can be loaded directly through `torchvision`. The weights are automatically downloaded and cached when the corresponding pretrained model is instantiated.

RadiologyNET-pretrained weights can be obtained from the [RadiologyNET-TL-models repository](https://github.com/AIlab-RITEH/RadiologyNET-TL-models). Please follow the instructions provided in the original repository for downloading and loading the corresponding weights.

### Training Configuration

| Setting | Value |
|:---|:---|
| Optimizer | AdamW |
| Loss function | Asymmetric Loss |
| Learning rate | 0.003 |
| Maximum epochs | 100 |
| Checkpoint saving | Every epoch |
| Model selection | Highest validation mF1 |
| HPC batch size | 16 |
| BraTS2018 batch size | 8 |
| 3DLSC-COVID batch size | 16 |

The model parameters were saved after each epoch, and the model achieving the highest macro-F1 (mF1) score on the validation set was retained.

The experiments reported in the manuscript were originally conducted using PyTorch 2.0. The current repository has also been tested with PyTorch 2.5.1, which is used in the provided environment configuration.


