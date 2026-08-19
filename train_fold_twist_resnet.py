import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchio as tio

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from collections import Counter
from tqdm import tqdm

from twist3dnet_with_resnet import Twist_ResNet_3D
from utils import AsymmetricLossOptimized, calculate_metrics
from Hpc_dataset import MyNiiDataset, ResampleSubsetSampler


def parse_args():
    parser = argparse.ArgumentParser(description='Train Twist3DNet on HPC dataset with five-fold cross-validation.')

    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory of the HPC dataset.')
    parser.add_argument('--output-dir', type=str, default='hpc_weights',
                        help='Directory for saving model weights and results.')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-resamples', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nifti-axis-order', type=str, default='HWD',
                        choices=['HWD', 'DHW'])

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def one_hot_labels(labels, num_classes=2):
    return F.one_hot(labels, num_classes=num_classes).float()


def calculate_class_accuracies(predictions, labels, num_classes=2):
    correct = torch.zeros(num_classes, device=predictions.device)
    total = torch.zeros(num_classes, device=predictions.device)

    for i in range(num_classes):
        total[i] = (labels == i).sum()
        correct[i] = ((predictions == i) & (labels == i)).sum()

    return correct / (total + 1e-8)


def train_epoch(model, data_loader, optimizer, loss_function, device):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(data_loader, file=sys.stdout, desc='Training', leave=False):
        images = images.float().to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)
        targets = one_hot_labels(labels, num_classes=2)

        optimizer.zero_grad(set_to_none=True)

        # Model output should be raw logits without Softmax
        outputs = model(images)
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(data_loader)


@torch.no_grad()
def evaluate_epoch(model, data_loader, loss_function, device):
    model.eval()

    running_loss = 0.0
    all_outputs = []
    all_labels = []

    for images, labels in tqdm(data_loader, file=sys.stdout, desc='Validation', leave=False):
        images = images.float().to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)
        targets = one_hot_labels(labels, num_classes=2)

        outputs = model(images)
        loss = loss_function(outputs, targets)

        running_loss += loss.item()
        all_outputs.append(outputs.cpu())
        all_labels.append(labels.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    predictions = torch.argmax(all_outputs, dim=1)

    val_loss = running_loss / len(data_loader)
    accuracy = (predictions == all_labels).float().mean().item()

    outputs_gpu = all_outputs.to(device)
    labels_gpu = all_labels.to(device)
    predictions_gpu = predictions.to(device)

    mF1, mAccuracy, mPrecision, mRecall, f1_per_class = calculate_metrics(
        outputs_gpu, labels_gpu
    )

    class_accuracies = calculate_class_accuracies(
        predictions_gpu, labels_gpu, num_classes=2
    )

    if torch.is_tensor(mF1):
        mF1 = mF1.item()
    if torch.is_tensor(mAccuracy):
        mAccuracy = mAccuracy.item()
    if torch.is_tensor(mPrecision):
        mPrecision = mPrecision.item()
    if torch.is_tensor(mRecall):
        mRecall = mRecall.item()

    if torch.is_tensor(f1_per_class):
        f1_per_class = f1_per_class.detach().cpu().tolist()

    class_accuracies = class_accuracies.detach().cpu().tolist()

    metrics = {
        'val_loss': float(val_loss),
        'accuracy': float(accuracy),
        'mF1': float(mF1),
        'mAccuracy': float(mAccuracy),
        'mPrecision': float(mPrecision),
        'mRecall': float(mRecall),
        'f1_per_class': [float(x) for x in f1_per_class],
        'class_accuracies': [float(x) for x in class_accuracies]
    }

    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)

    target_shape = (1, 13, 256, 256)

    # Online augmentation for training
    train_transform = tio.Compose([
        tio.RandomFlip(
            axes=(0, 1),
            flip_probability=0.5
        ),
        tio.RandomAffine(
            scales=(0.95, 1.05),
            degrees=(0, 0, 5),
            translation=(5, 5, 0),
            image_interpolation='linear'
        )
    ])

    # Training and validation use the same images, but only training
    # data receive random augmentation.
    train_dataset = MyNiiDataset(
        root_dir=args.data_root,
        transform=train_transform,
        step='train',
        target_shape=target_shape,
        nifti_axis_order=args.nifti_axis_order
    )

    val_dataset = MyNiiDataset(
        root_dir=args.data_root,
        transform=None,
        step='train',
        target_shape=target_shape,
        nifti_axis_order=args.nifti_axis_order
    )

    if train_dataset.images != val_dataset.images:
        raise RuntimeError('Training and validation dataset ordering does not match.')

    print('Class mapping:', train_dataset.class_to_idx)

    # Dataset statistics
    labels = np.asarray(train_dataset.targets, dtype=np.int64)
    counts = Counter(labels.tolist())
    total = len(labels)

    print('Global label distribution:')
    for class_idx in sorted(counts.keys()):
        print(f'Class {class_idx}: {counts[class_idx]} ({counts[class_idx] / total:.2%})')

    # Five-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    fold_best_results = []

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(np.arange(len(train_dataset))), start=1
    ):
        print(f'\n{"=" * 60}')
        print(f'Fold {fold}/5')
        print(f'{"=" * 60}')
        print(f'Training samples: {len(train_idx)}')
        print(f'Validation samples: {len(val_idx)}')

        # HPC training dataset undergoes 15 rounds of resampling
        train_sampler = ResampleSubsetSampler(
            train_idx,
            num_resamples=args.num_resamples,
            seed=args.seed + fold
        )

        val_sampler = SubsetRandomSampler(val_idx.tolist())

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        model = Twist_ResNet_3D().to(device)


        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Asymmetric Loss used in the experiments
        loss_function = AsymmetricLossOptimized()

        # Each fold must have its own best mF1
        best_mF1 = -float('inf')
        best_metrics = None

        fold_dir = os.path.join(args.output_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                loss_function,
                device
            )

            metrics = evaluate_epoch(
                model,
                val_loader,
                loss_function,
                device
            )

            print(f'Epoch [{epoch}/{args.epochs}]')
            print(
                f'Train Loss: {train_loss:.5f} | '
                f'Validation Loss: {metrics["val_loss"]:.5f}'
            )
            print(
                f'Accuracy: {metrics["accuracy"]:.5f} | '
                f'mF1: {metrics["mF1"]:.5f} | '
                f'mAccuracy: {metrics["mAccuracy"]:.5f} | '
                f'mPrecision: {metrics["mPrecision"]:.5f} | '
                f'mRecall: {metrics["mRecall"]:.5f}'
            )

            for i, acc in enumerate(metrics['class_accuracies']):
                print(f'Class {i} Accuracy: {acc:.5f}')

            # Save model parameters after every epoch
            epoch_path = os.path.join(fold_dir, f'epoch_{epoch:03d}.pth')
            torch.save(model.state_dict(), epoch_path)

            # Retain the model with the highest validation mF1 in each fold
            if metrics['mF1'] > best_mF1:
                best_mF1 = metrics['mF1']

                best_metrics = {
                    'fold': fold,
                    'epoch': epoch,
                    **metrics
                }

                best_path = os.path.join(fold_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_path)

                print(f' -> New best model saved at {best_path}')

        # Save best metrics for this fold
        fold_best_results.append(best_metrics)

        with open(
            os.path.join(fold_dir, 'best_metrics.json'),
            'w',
            encoding='utf-8'
        ) as f:
            json.dump(best_metrics, f, indent=4)

        print(f'Best result of Fold {fold}:')
        print(
            f'Epoch: {best_metrics["epoch"]} | '
            f'mF1: {best_metrics["mF1"]:.5f} | '
            f'Accuracy: {best_metrics["accuracy"]:.5f}'
        )

    # Average performance across five folds
    metric_names = [
        'accuracy',
        'mF1',
        'mAccuracy',
        'mPrecision',
        'mRecall'
    ]

    summary = {
        'folds': fold_best_results,
        'mean': {},
        'std': {}
    }

    print(f'\n{"=" * 60}')
    print('Five-fold Cross-Validation Summary')
    print(f'{"=" * 60}')

    for metric_name in metric_names:
        values = np.asarray(
            [result[metric_name] for result in fold_best_results],
            dtype=np.float64
        )

        mean_value = values.mean()
        std_value = values.std(ddof=0)

        summary['mean'][metric_name] = float(mean_value)
        summary['std'][metric_name] = float(std_value)

        print(f'{metric_name}: {mean_value:.5f} ± {std_value:.5f}')

    summary_path = os.path.join(
        args.output_dir,
        'five_fold_summary.json'
    )

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f'\nSummary saved to: {summary_path}')
    print('Finished Training with Five-Fold Cross-Validation')


if __name__ == '__main__':
    main()
