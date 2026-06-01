import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import math

class Config:
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    RANDOM_SEED = 42
    
    DATASET_PATH = os.path.expanduser('~/datasetp')
    if not os.path.exists(DATASET_PATH):
        current_dataset = os.path.join(os.getcwd(), 'dataset')
        if os.path.exists(current_dataset):
            DATASET_PATH = current_dataset
    
    CHECKPOINT_DIR = 'checkpoints'
    MODEL_PATH = 'localization_model.pth'
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    WEIGHT_DECAY = 1e-5


class AugmentedLocalizationDataset(Dataset):
    def __init__(self, dataset_path, augment=True):
        self.dataset_path = Path(dataset_path)
        self.augment = augment
        self.samples = []
        
        for dir_path in sorted(self.dataset_path.iterdir()):
            if dir_path.is_dir() and dir_path.name.isdigit():
                input_dir = dir_path / 'input'
                output_dir = dir_path / 'output'
                
                map_file = input_dir / 'map.npz'
                scan_file = input_dir / 'scan.npz'
                pose_file = output_dir / 'pose.txt'
                
                if map_file.exists() and scan_file.exists() and pose_file.exists():
                    self.samples.append({
                        'map': map_file,
                        'scan': scan_file,
                        'pose': pose_file,
                    })
        
        print(f"{len(self.samples)} samples loaded")
        if augment:
            print(f"{len(self.samples) * 5} samples will be generated")
    
    def __len__(self):
        if self.augment:
            return len(self.samples) * 5
        return len(self.samples)
    
    def _augment_pose(self, pose):
        x, y, sin_angle, cos_angle = pose
        
        angle = np.arctan2(sin_angle, cos_angle)
        rot_deg = np.random.uniform(-20, 20)
        rot_rad = np.radians(rot_deg)
        new_angle = angle + rot_rad
        
        new_x = x + np.random.uniform(-1, 1)
        new_y = y + np.random.uniform(-1, 1)
        
        return np.array([new_x, new_y, np.sin(new_angle), np.cos(new_angle)], dtype=np.float32)
    
    def _augment_image(self, image):
        if np.random.random() > 0.5:
            image = np.fliplr(image).copy()
        
        if np.random.random() > 0.5:
            shift = np.random.randint(-5, 5)
            image = np.roll(image, shift, axis=1)
        
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 0.01, image.shape).astype(np.float32)
            image = image + noise
            image = np.clip(image, 0, 1)
        
        return image
    
    def __getitem__(self, idx):
        if self.augment:
            original_idx = idx % len(self.samples)
            aug_idx = idx // len(self.samples)
        else:
            original_idx = idx
            aug_idx = 0
        
        sample = self.samples[original_idx]
        
        map_data = np.load(sample['map'])
        map_matrix = map_data['data'] if 'data' in map_data else map_data['arr_0']
        map_matrix = map_matrix.astype(np.float32)
        if map_matrix.max() > 0:
            map_matrix = map_matrix / map_matrix.max()
        
        scan_data = np.load(sample['scan'])
        scan_matrix = scan_data['data'] if 'data' in scan_data else scan_data['arr_0']
        scan_matrix = scan_matrix.astype(np.float32)
        if scan_matrix.max() > 0:
            scan_matrix = scan_matrix / scan_matrix.max()
        
        pose = np.loadtxt(sample['pose'], dtype=np.float32)
        
        if self.augment and aug_idx > 0:
            pose = self._augment_pose(pose)
            map_matrix = self._augment_image(map_matrix)
            scan_matrix = self._augment_image(scan_matrix)
        
        if len(map_matrix.shape) == 2:
            map_tensor = torch.from_numpy(map_matrix).unsqueeze(0)
            scan_tensor = torch.from_numpy(scan_matrix).unsqueeze(0)
        else:
            map_tensor = torch.from_numpy(map_matrix)
            scan_tensor = torch.from_numpy(scan_matrix)
        
        input_tensor = torch.cat([map_tensor, scan_tensor], dim=0)
        target = torch.from_numpy(pose)
        
        return input_tensor, target


class SimpleLocalizationCNN(nn.Module):
    def __init__(self, input_channels=2):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1: 2 -> 16
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Block 2: 16 -> 32
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 3: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 4: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 5: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # MSE
        mse_loss = self.mse(pred, target)
        
        # sin²+cos²=1
        pred_norm = torch.norm(pred[:, 2:], dim=1)
        norm_loss = F.mse_loss(pred_norm, torch.ones_like(pred_norm))
        
        # angle
        pred_angle = torch.atan2(pred[:, 2], pred[:, 3])
        target_angle = torch.atan2(target[:, 2], target[:, 3])
        angle_diff = torch.abs(pred_angle - target_angle)
        angle_diff = torch.min(angle_diff, 2 * math.pi - angle_diff)
        angle_loss = angle_diff.mean()
        
        total_loss = mse_loss + 5.0 * norm_loss + 1.0 * angle_loss
        
        return total_loss, mse_loss, norm_loss, angle_loss


def calculate_metrics(y_true, y_pred):
    pos_error = np.sqrt((y_true[:, 0] - y_pred[:, 0])**2 + 
                         (y_true[:, 1] - y_pred[:, 1])**2)
    
    angle_true = np.arctan2(y_true[:, 2], y_true[:, 3])
    angle_pred = np.arctan2(y_pred[:, 2], y_pred[:, 3])
    angle_diff = np.abs(angle_true - angle_pred)
    angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
    angle_error_deg = np.degrees(angle_diff)
    
    pred_norm = np.sqrt(y_pred[:, 2]**2 + y_pred[:, 3]**2)
    
    return {
        'pos_error_mean': float(np.mean(pos_error)),
        'pos_error_median': float(np.median(pos_error)),
        'angle_error_mean_deg': float(np.mean(angle_error_deg)),
        'angle_error_median_deg': float(np.median(angle_error_deg)),
        'pred_norm_mean': float(np.mean(pred_norm)),
        'good_positions': float(np.mean(pos_error < 0.5) * 100),
    }


def train_model(model, train_loader, val_loader, config):
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, 
                          weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=20,
                                                      min_lr=1e-6)
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Device {config.DEVICE}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'pos_error': [], 'angle_error': [], 'norm': []}
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_loss_mse = 0
        train_loss_norm = 0
        train_loss_angle = 0
        
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss, loss_mse, loss_norm, loss_angle = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_loss_mse += loss_mse.item()
            train_loss_norm += loss_norm.item()
            train_loss_angle += loss_angle.item()
        
        n_train = len(train_loader)
        train_loss /= n_train
        train_loss_mse /= n_train
        train_loss_norm /= n_train
        train_loss_angle /= n_train
        
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
                outputs = model(inputs)
                loss, _, _, _ = criterion(outputs, targets)
                val_loss += loss.item()
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        metrics = calculate_metrics(all_targets, all_preds)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['pos_error'].append(metrics['pos_error_mean'])
        history['angle_error'].append(metrics['angle_error_mean_deg'])
        history['norm'].append(metrics['pred_norm_mean'])
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"Loss: train={train_loss:.4f}, val={val_loss:.4f}")
        print(f"  - MSE: {train_loss_mse:.4f}, Norm: {train_loss_norm:.4f}, Angle: {train_loss_angle:.4f}")
        print(f"\nMetrics:")
        print(f"  Position Error: mean={metrics['pos_error_mean']:.3f}m, median={metrics['pos_error_median']:.3f}m")
        print(f"  Angle Error: {metrics['angle_error_mean_deg']:.1f}°, median={metrics['angle_error_median_deg']:.1f}°")
        print(f"  Sin/Cos Norm: {metrics['pred_norm_mean']:.3f}")
        print(f"  Good positions (<0.5m): {metrics['good_positions']:.1f}%")
        print(f"{'='*50}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
            print(f"Best model saved (val_loss={best_val_loss:.4f})")
        
        scheduler.step(val_loss)
        
        if metrics['pos_error_mean'] < 0.3 and metrics['angle_error_mean_deg'] < 10:
            print(f"\nTarget accuracy achieved!")
            break
    
    best_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))
    
    return model, history


def plot_results(history):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0, 0].plot(epochs, history['train_loss'], label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs, history['pos_error'], color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Error (m)')
    axes[0, 1].set_title('Mean Position Error')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(epochs, history['angle_error'], color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Error (deg)')
    axes[1, 0].set_title('Mean Angle Error')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(epochs, history['norm'], color='orange')
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='Target')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Norm')
    axes[1, 1].set_title('Sin/Cos Norm')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.close()
    print("Train visualization saved to training_history.png")


def main():
    print("=" * 60)
    print("CNN model")
    print("=" * 60)
    
    config = Config()
    
    if not os.path.exists(config.DATASET_PATH):
        print(f"Error: no dataset: {config.DATASET_PATH}")
        sys.exit(1)
    
    print(f"\nDevice: {config.DEVICE}")
    
    dataset = AugmentedLocalizationDataset(config.DATASET_PATH, augment=True)
    
    if len(dataset) == 0:
        print("Error: no data!")
        sys.exit(1)
    
    sample_input, _ = dataset[0]
    print(f"Input data size: {sample_input.shape}")
    
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=config.VALIDATION_SPLIT, random_state=config.RANDOM_SEED
    )
    
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    model = SimpleLocalizationCNN(input_channels=2).to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    model, history = train_model(model, train_loader, val_loader, config)
    
    plot_results(history)
    
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"\nModel saved to {config.MODEL_PATH}")
    
    print("\n" + "=" * 60)
    print("Train completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()