"""
Script huấn luyện mô hình CNN sử dụng PyTorch
Thay thế hoàn toàn TensorFlow/Keras
"""
import os
import sys
import logging
import traceback
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Kiểm tra PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    logger.info(f"✓ PyTorch version: {torch.__version__}")
    logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError:
    logger.error("✗ PyTorch không được cài đặt!")
    logger.error("Cài đặt: pip install torch torchvision")
    sys.exit(1)

# Đường dẫn đến dữ liệu
DATA_DIR = 'data/kagglecatsanddogs_3367a/PetImages'
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info(f"Sử dụng device: {DEVICE}")


class CatDogDataset(Dataset):
    """Dataset class cho Cat và Dog"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            logger.warning(f"Không thể load ảnh {img_path}: {str(e)}")
            # Trả về ảnh đen nếu lỗi
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)


class CatDogCNN(nn.Module):
    """Mô hình CNN cho phân loại Cat và Dog"""
    def __init__(self):
        super(CatDogCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Block 4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Classification head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 9 * 9, 512)  # 150/2^4 = 9.375 -> 9
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 1)
        
    def forward(self, x):
        # Block 1
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        
        # Block 4
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        
        # Classification
        x = self.flatten(x)
        x = self.dropout1(torch.relu(self.bn5(self.fc1(x))))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x


def load_image_paths(directory, label):
    """Load danh sách đường dẫn ảnh từ thư mục"""
    image_paths = []
    labels = []
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(directory, ext)))
        image_paths.extend(glob.glob(os.path.join(directory, ext.upper())))
    
    labels = [label] * len(image_paths)
    
    return image_paths, labels


def check_data_directory():
    """Kiểm tra thư mục dữ liệu"""
    if not os.path.exists(DATA_DIR):
        logger.error(f"Thư mục dữ liệu không tồn tại: {DATA_DIR}")
        return False
    
    cat_dir = os.path.join(DATA_DIR, 'Cat')
    dog_dir = os.path.join(DATA_DIR, 'Dog')
    
    if not os.path.exists(cat_dir) or not os.path.exists(dog_dir):
        logger.error("Thư mục Cat hoặc Dog không tồn tại")
        return False
    
    return True


def prepare_data():
    """Chuẩn bị dữ liệu"""
    cat_dir = os.path.join(DATA_DIR, 'Cat')
    dog_dir = os.path.join(DATA_DIR, 'Dog')
    
    logger.info("Đang load đường dẫn ảnh...")
    cat_paths, cat_labels = load_image_paths(cat_dir, 0)
    dog_paths, dog_labels = load_image_paths(dog_dir, 1)
    
    all_paths = cat_paths + dog_paths
    all_labels = cat_labels + dog_labels
    
    logger.info(f"Tổng số ảnh: {len(all_paths)}")
    logger.info(f"  - Cat: {len(cat_paths)}")
    logger.info(f"  - Dog: {len(dog_paths)}")
    
    # Split train/validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    logger.info(f"Training samples: {len(train_paths)}")
    logger.info(f"Validation samples: {len(val_paths)}")
    
    return train_paths, val_paths, train_labels, val_labels


def train():
    """Huấn luyện mô hình"""
    try:
        # Kiểm tra dữ liệu
        if not check_data_directory():
            logger.error("Dữ liệu không hợp lệ")
            return None
        
        # Chuẩn bị dữ liệu
        train_paths, val_paths, train_labels, val_labels = prepare_data()
        
        # Data transforms
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(40),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Tạo datasets
        train_dataset = CatDogDataset(train_paths, train_labels, transform=train_transform)
        val_dataset = CatDogDataset(val_paths, val_labels, transform=val_transform)
        
        # Tạo dataloaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        # Tạo mô hình
        model = CatDogCNN().to(DEVICE)
        logger.info("Mô hình đã được tạo:")
        logger.info(str(model))
        
        # Loss và optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)
        
        # Training
        best_val_acc = 0.0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        os.makedirs('models', exist_ok=True)
        
        logger.info("Bắt đầu huấn luyện...")
        logger.info(f"Device: {DEVICE}, Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
        
        for epoch in range(EPOCHS):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images).squeeze()
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            scheduler.step(val_loss)
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            logger.info(f"Epoch [{epoch+1}/{EPOCHS}]")
            logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'models/cat_dog_model_pytorch.pth')
                logger.info(f"  → Đã lưu model tốt nhất (Val Acc: {val_acc:.2f}%)")
        
        # Lưu model cuối cùng
        torch.save(model.state_dict(), 'models/cat_dog_model_pytorch_final.pth')
        torch.save(model, 'models/cat_dog_model_pytorch_full.pth')  # Save full model để dễ load
        
        # Vẽ biểu đồ
        plot_training_history(train_losses, train_accs, val_losses, val_accs)
        
        logger.info("=" * 50)
        logger.info("Huấn luyện hoàn tất!")
        logger.info(f"Độ chính xác tốt nhất (validation): {best_val_acc:.2f}%")
        logger.info("=" * 50)
        
        return model
        
    except KeyboardInterrupt:
        logger.warning("\nHuấn luyện bị gián đoạn")
        return None
    except Exception as e:
        logger.error(f"Lỗi khi huấn luyện: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """Vẽ biểu đồ quá trình huấn luyện"""
    try:
        epochs = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(14, 5))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_accs, 'bo-', label='Training accuracy', linewidth=2, markersize=6)
        plt.plot(epochs, val_accs, 'ro-', label='Validation accuracy', linewidth=2, markersize=6)
        plt.title('Model Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_losses, 'bo-', label='Training loss', linewidth=2, markersize=6)
        plt.plot(epochs, val_losses, 'ro-', label='Validation loss', linewidth=2, markersize=6)
        plt.title('Model Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/training_history_pytorch.png', dpi=300, bbox_inches='tight')
        logger.info("Đã lưu biểu đồ vào models/training_history_pytorch.png")
    except Exception as e:
        logger.error(f"Lỗi khi vẽ biểu đồ: {str(e)}")


if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Bắt đầu huấn luyện mô hình Cat/Dog Classifier với PyTorch")
    logger.info("=" * 50)
    
    try:
        model = train()
        if model is not None:
            logger.info("Huấn luyện thành công!")
        else:
            logger.error("Huấn luyện thất bại!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Lỗi không mong muốn: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

