import os
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from scipy.ndimage import zoom
from tqdm import tqdm
import wandb
import argparse
import numpy as np

# Initialize W&B
wandb.init(project="cpsc-8650-courseproject", entity="oxxocodes")

# Configuration
config = wandb.config

class CustomDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.nii')]
        self.transform = transform
        self.labels_map = {row[0]: (row[1], row[2]) for row in pd.read_csv(labels_file).itertuples(index=False)}
        self.data = []

        print("Loading data...")
        for idx in tqdm(range(len(self.files)), total=len(self.files)):
            img_name = os.path.join(self.root_dir, self.files[idx])
            image = nib.load(img_name).get_fdata()

            image = zoom(image, (config.zoom_factor, config.zoom_factor, config.zoom_factor)) # Reduce 3D scan to be 25% of original size
            image = (image - image.mean()) / image.std() # Z-score normaliztion
            
            original_image = torch.tensor(image, dtype=torch.float32)
            original_image = original_image.reshape(-1)

            flipped_image = np.flip(image, axis=0).copy()  # Horizontal flip
            flipped_image = torch.tensor(flipped_image, dtype=torch.float32)
            flipped_image = flipped_image.reshape(-1)  # Flatten

            img_id = f"smwp1_{self.files[idx].split('.')[0].split('smwp1')[1]}".split("_T1")[0] # Fix filename->CSV row ids
            targets = self.labels_map[img_id]
            targets = torch.tensor(targets, dtype=torch.float32)

            self.data.append((original_image, targets))
            self.data.append((flipped_image, targets))
        print("Data loaded")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]

class ImageSliceDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels_map = {row[0]: (row[1], row[2]) for row in pd.read_csv(labels_file).itertuples(index=False)}
        self.data = []
        self.labels = []

        print("Loading data...")
        for f in tqdm(os.listdir(root_dir)):
            if f.endswith('.nii'):
                img_name = os.path.join(self.root_dir, f)
                image = nib.load(img_name).get_fdata()
                
                img_id = f"smwp1_{f.split('.')[0].split('smwp1')[1]}".split("_T1")[0]
                targets = self.labels_map[img_id]
                targets = torch.tensor(targets, dtype=torch.float32)

                for slice_idx in range(image.shape[2]):  # Iterate over each slice
                    slice_img = image[:, :, slice_idx]
                    slice_img = (slice_img - slice_img.mean()) / slice_img.std()  # Z-score normalization

                    # Original slice processing
                    original_img = torch.tensor(slice_img, dtype=torch.float32)
                    original_img = original_img.reshape(-1)  # Flatten the slice

                    # Horizontally flipped slice processing
                    flipped_img = slice_img[:, ::-1].copy()
                    flipped_img = torch.tensor(flipped_img, dtype=torch.float32)
                    flipped_img = flipped_img.reshape(-1)  # Flatten the slice

                    if self.transform:
                        original_img = self.transform(original_img)
                        flipped_img = self.transform(flipped_img)

                    # Append both original and flipped images to the dataset
                    self.data.append(original_img)
                    self.labels.append(targets)
                    self.data.append(flipped_img)
                    self.labels.append(targets)
        print("Data loaded")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SimpleMLP(nn.Module):
    def __init__(self, input_size, dropout_p):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)

recent_val_losses = []

def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    sum_abs_errors = 0  # Sum of absolute errors for MAE
    count_samples = 0   # Total number of samples

    outputs = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            outputs.append(output)
            loss = nn.MSELoss(reduction='sum')(output, target)  # Sum the losses for precise average calculation
            val_loss += loss.item()

            # Calculate absolute errors
            abs_errors = torch.abs(output - target)
            sum_abs_errors += abs_errors.sum().item()
            count_samples += target.numel()  # Number of elements in target batch

    all_outputs = torch.cat(outputs, dim=0)
    std = all_outputs.std(dim=0)
    std_lower_threshold = std[0].item()
    std_upper_threshold = std[1].item()

    # Calculate average MSE
    avg_mse = val_loss / count_samples
    # Calculate RMSE
    rmse = torch.sqrt(torch.tensor(avg_mse))
    # Calculate MAE
    mae = sum_abs_errors / count_samples

    print(f'\nValidation set: Average MSE: {avg_mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}')

    # Update the list of recent validation losses
    recent_val_losses.append(avg_mse)
    
    # Ensure the list doesn't grow beyond 10 elements
    if len(recent_val_losses) > 10:
        recent_val_losses.pop(0)
    
    # Compute the moving average of validation losses
    moving_avg_val_loss = sum(recent_val_losses) / len(recent_val_losses)
    print(f'Moving Average Validation Loss (last 10 MSE): {moving_avg_val_loss:.4f}')
    print(f"Val RMSE: {rmse.item():.4f}")
    print(f"Val MAE: {mae:.4f}")
    
    # Log the metrics to W&B
    wandb.log({
        "val_mse": avg_mse,
        "val_rmse": rmse.item(),
        "val_mae": mae,
        "val_mov_avg_mse": moving_avg_val_loss,
        "val_std_lower_threshold": std_lower_threshold,
        "val_std_upper_threshold": std_upper_threshold
    })

def train(model, device, train_loader, val_loader, optimizer, scheduler):
    idx = 0
    for epoch in range(1, config.epochs+1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if idx % config.log_iter == 0 and idx != 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.2f}')
                wandb.log({"train_loss": loss.item(), "epoch": epoch})

            if idx % config.eval_iter == 0:
                validate(model, device, val_loader)
            
            idx += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--min_lr_divisor', type=int, default=3)
    parser.add_argument('--dropout_p', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--validation_split', type=float, default=0.1)
    parser.add_argument('--log_iter', type=int, default=1)
    parser.add_argument('--zoom_factor', type=float, default=0.25)
    args = parser.parse_args()

    # Use args.learning_rate and args.batch_size where appropriate
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.min_lr_divisor = args.min_lr_divisor
    config.dropout_p = args.dropout_p
    config.epochs = args.epochs
    config.validation_split = args.validation_split
    config.log_iter = args.log_iter
    config.zoom_factor = args.zoom_factor

    torch.manual_seed(42)
    data_dir = '/scratch/nbrown9/cpsc-8650/n171_smwp1'
    dataset = CustomDataset(root_dir=data_dir, labels_file="PTs_500_4k_blinded.csv")
    # dataset = ImageSliceDataset(root_dir=data_dir, labels_file="PTs_500_4k_blinded.csv")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # config.eval_iter = len(train_loader)*config.epochs // 10
    config.eval_iter = 10
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda:0"
    print(f"Training on {device}")
    
    first_sample_tensor, _ = dataset[0]
    input_size = first_sample_tensor.shape[0]
    print(f"Input size: {input_size}")
    
    model = SimpleMLP(input_size, config.dropout_p).to(device)
    wandb.watch(model, log="all")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} parameters")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader)*config.epochs, eta_min=config.learning_rate/config.min_lr_divisor)
    
    train(model, device, train_loader, val_loader, optimizer, scheduler)

if __name__ == '__main__':
    main()
    wandb.finish()
