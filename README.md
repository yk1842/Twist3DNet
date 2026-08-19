# Twist3DNet
Twist3DNet is a 2D–3D hybrid network based on transfer learning for prognosis classification of hypopharyngeal cancer. The MMPreTrain-based configuration provided in this repository was developed for the HPC dataset. The checkpoint specified in the configuration corresponds to pretrained ResNet18-3D weights used for backbone initialization rather than a trained Twist3DNet checkpoint.


We have updated the repository with the latest implementation of Twist3DNet for hypopharyngeal cancer prognosis classification. In addition to the original 2D–3D hybrid framework based on MMPreTrain, this update provides the simple model code for Twist3DNet + ResNet18-2D, as well as additional 2D branch implementations including ShuffleNetV2-2D, SENet18-2D, and ConvMixer-2D. Furthermore, two modified hybrid networks, H-DenseUNet for classification and MDU-Net for classification, are included as comparative models to facilitate fair performance evaluation against the proposed Twist3DNet framework.



## Table of Contents

- [Environment Dependencies](#environment-dependencies)
- [Data Preparation](#data-preparation)
  - [HPC Dataset (Private)](#hpc-dataset-private)
  - [BraTS2018 Dataset](#brats2018-dataset)
  - [3DLSC-COVID Dataset](#3dlsc-covid-dataset)
- [Usage](#usage)
- [Results](#results)
- [Additional Documentation](#additional-documentation)


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

> **Note:** System-level dependencies (e.g., `bzip2`, `libgcc`, `openssl`, and `zlib`) are omitted from the list above. Additional system packages may be required depending on the operating system and CUDA environment.

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

The original dataset can be obtained from [MICCAI BraTS 2018: Data](https://www.med.upenn.edu/sbia/brats2018/). Please follow the official BraTS instructions to obtain the dataset before running the preprocessing and training procedures.


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

The publicly available data can be obtained from the [DeepSC-COVID GitHub repository](https://github.com/XiaofeiWang2018/DeepSC-COVID). Please follow the instructions provided in the original repository to download the released 3DLSC-COVID data.


## Usage

After installing the required dependencies, prepare the datasets according to the procedures described in the [Data Preparation](#data-preparation) section.

### Model Implementation

The main implementation of Twist3DNet with the ResNet18-2D branch is provided in:

```text
twist3dnet_with_resnet.py
```

The default 2D branch is **ResNet18-2D**. Alternative 2D branches, including **ShuffleNetV2-2D**, **SENet18-2D**, and **ConvMixer-2D**, are provided in the corresponding 2D branch implementation file.

Users can replace the default ResNet18-2D branch in `twist3dnet_with_resnet.py` with the corresponding implementation according to their experimental requirements.

The repository also provides implementations of several comparative models used in this study, including ResNet18-3D, ShuffleNetV2-3D, SENet18-3D, ConvMixer-3D, H-DenseUNet for classification, and MDU-Net for classification.

### Training

The five-fold cross-validation training implementation used for Twist3DNet is provided in:

```text
train_fold_twist.py
```

The training script includes the main training procedure, data augmentation, cross-validation, checkpoint saving, and model selection strategy used in our experiments.

The dataset loading and preprocessing implementation used for the HPC experiments is provided in:

```text
Hpc_dataset.py
```

Because the HPC dataset is private and cannot be publicly distributed, users who wish to adapt the training pipeline to their own datasets should modify the dataset loading and path configuration accordingly.

### Pre-trained Weights

Transfer learning is used to initialize the feature extraction branches.

For standard 2D backbones, **ImageNet-pretrained weights** can be loaded directly through `torchvision`. The corresponding weights are automatically downloaded and cached when a pretrained model is instantiated.

**RadiologyNET-pretrained weights** can be obtained from the [RadiologyNET-TL-models repository](https://github.com/AIlab-RITEH/RadiologyNET-TL-models). Please follow the instructions provided in the original repository to download and load the corresponding pretrained weights.

For the 3D branch, the checkpoint used for initialization corresponds to standard pretrained **ResNet18-3D** weights. These pretrained weights are used only for backbone initialization and should not be confused with the final trained Twist3DNet checkpoints.

The pretrained-weight paths or initialization settings should be configured in `twist3dnet_with_resnet.py` according to the selected 2D and 3D backbones.

### Training Configuration

The main training settings used in our experiments are summarized below.

| Setting                | Value                  |
| :--------------------- | :--------------------- |
| Optimizer              | AdamW                  |
| Loss function          | Asymmetric Loss        |
| Learning rate          | 0.003                  |
| Weight decay           | 1e-4                   |
| Maximum epochs         | 100                    |
| Random seed            | 42                     |
| Checkpoint saving      | Every epoch            |
| Model selection        | Highest validation mF1 |
| HPC input size         | 13×256×256             |
| HPC resampling         | 15 rounds              |
| HPC batch size         | 16                     |
| BraTS2018 batch size   | 8                      |
| 3DLSC-COVID resampling | 2 rounds               |
| 3DLSC-COVID batch size | 16                     |

For the HPC experiments, model parameters were saved after each epoch. Within each cross-validation fold, the checkpoint achieving the highest **macro-F1 (mF1)** score on the validation subset was retained as the best model. The final performance was obtained by averaging the evaluation metrics across the five folds.

### Experimental Environment

The experiments reported in the manuscript were conducted on a server equipped with one **AMD EPYC 7763 64-core/128-thread CPU** and four **NVIDIA GeForce RTX 4090 GPUs** with a total of 96 GB GPU memory, running Ubuntu 22.04 LTS.

The original experiments were conducted using **PyTorch 2.0**. The current repository has also been tested with **PyTorch 2.5.1**, which is used in the provided environment configuration.


## Results

### Results on the HPC Dataset

The following table summarizes the performance of Twist3DNet and the comparative models on the HPC dataset. Classification metrics are reported as **mean ± standard deviation** over five-fold cross-validation.

| Method                       |          mF1 (%) |    mAccuracy (%) |   mPrecision (%) |      mRecall (%) | Params (M) |   GFLOPs | Latency (ms) | Peak Memory (GB) |
| :--------------------------- | ---------------: | ---------------: | ---------------: | ---------------: | ---------: | -------: | -----------: | ---------------: |
| ResNet18-3D                  |     64.25 ± 7.57 |     66.38 ± 9.15 |     64.81 ± 6.23 |     66.00 ± 9.02 |      33.18 |    95.24 |        21.57 |            0.243 |
| ShuffleNetV2-3D              |     65.87 ± 9.06 |     67.17 ± 8.30 |     66.57 ± 8.57 |     66.94 ± 8.26 |   **1.30** | **2.02** |        10.14 |        **0.210** |
| SENet18-3D                   |     65.96 ± 5.51 |     67.72 ± 4.84 |     68.25 ± 6.65 |     67.43 ± 5.97 |      33.27 |    18.71 |     **6.87** |            0.326 |
| ConvMixer-3D                 |     58.70 ± 8.83 |     60.47 ± 8.71 |     59.11 ± 9.19 |     59.46 ± 9.52 |      23.75 |   292.84 |       118.82 |            0.596 |
| Twist3DNet + ResNet18-2D     | **75.78 ± 6.36** | **78.12 ± 8.18** |     78.92 ± 8.09 | **75.30 ± 6.22** |      14.33 |    38.47 |        16.14 |            0.557 |
| Twist3DNet + ShuffleNetV2-2D |     69.87 ± 8.78 |    72.28 ± 10.69 |     72.43 ± 9.41 |     69.57 ± 7.67 |       7.81 |    11.87 |        20.71 |            0.523 |
| Twist3DNet + SENet18-2D      |     69.52 ± 5.29 |     74.75 ± 6.09 | **81.23 ± 4.67** |     70.31 ± 3.88 |      14.42 |    38.49 |        17.69 |            0.640 |
| Twist3DNet + ConvMixer-2D    |     63.87 ± 6.96 |     66.38 ± 8.35 |     64.35 ± 5.55 |     65.52 ± 8.15 |       5.51 |    15.00 |        15.03 |            0.661 |
| H-DenseUNet (classification) |     70.01 ± 8.68 |     72.25 ± 9.41 |     70.89 ± 6.96 |    71.84 ± 11.12 |      13.59 |   121.69 |        52.12 |            0.783 |
| MDU-Net (classification)     |    63.85 ± 10.76 |     68.88 ± 6.90 |     75.33 ± 5.82 |     68.13 ± 9.08 |      13.20 |   136.00 |        57.65 |            1.415 |

Among the evaluated models, **Twist3DNet + ResNet18-2D** achieved the highest mF1, mAccuracy, and mRecall, while **Twist3DNet + SENet18-2D** achieved the highest mPrecision. These results demonstrate the effectiveness of combining 2D feature extraction with 3D volumetric modeling for HPC prognosis classification.

## Additional Documentation

Additional details on the architectural adaptation of **H-DenseUNet** and **MDU-Net** from segmentation to classification, together with the corresponding objective-function ablation experiments, are provided in the following supplementary document:

[Architectural Adaptation and Objective-Function Ablation of H-DenseUNet and MDU-Net for Classification](Architectural%20Adaptation%20and%20Objective-Function%20Ablation%20of%20H-DenseUNet%20and%20MDU-Net%20for%20Classification.docx)


