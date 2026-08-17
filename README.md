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
git clone 
cd your-repo

# Create and activate conda environment
conda create -n twistnet python=3.12
conda activate twistnet

# Install dependencies
pip install -r requirements.txt
