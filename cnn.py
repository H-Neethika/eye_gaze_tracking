# eye_tracking_cnn.py
# Train on two-eye crops (50x200) saved as "<x>.<y>.<idx>.jpg".
# Labels are normalized to [0,1]. Use same preprocessing in live.

# --- Windows DPI awareness (ensures real pixel coords) ---
import sys
if sys.platform.startswith("win"):
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import os, sys as _sys, random
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pyautogui

def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

W, H = pyautogui.size()
print(f"[INFO] Screen: {W}x{H}")

class EyeGazeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels = [], []
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset folder not found: {root_dir}")
        for fn in os.listdir(root_dir):
            if not fn.lower().endswith(".jpg"): continue
            parts = fn.split(".")
            if len(parts) < 4: continue
            try:
                x = float(parts[0]); y = float(parts[1])
            except ValueError:
                continue
            self.image_paths.append(os.path.join(root_dir, fn))
            nx = max(0.0, min(1.0, x / float(W)))
            ny = max(0.0, min(1.0, y / float(H)))
            self.labels.append((nx, ny))
        if not self.image_paths:
            raise RuntimeError(f"No images found in {root_dir}. Expect '<x>.<y>.<idx>.jpg'.")
        print(f"[INFO] Loaded {len(self.image_paths)} images")

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("L")     # 50x200 grayscale saved by collector
        if self.transform: img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

transform = transforms.Compose([
    transforms.Resize((50, 200)),              # <- width 200 (two eyes)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

class EyeGazeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.pool  = nn.MaxPool2d(2, 2, 0)
        # (1,50,200) -> (32,25,100) -> (64,12,50)
        self.fc1   = nn.Linear(64*12*50, 512)
        self.fc2   = nn.Linear(512, 128)
        self.fc3   = nn.Linear(128, 2)
        self.out   = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(self.fc3(x))           # normalized [0,1]

def train(
    root_dir="eye_images",
    batch_size=64,
    num_epochs=25,
    lr=1e-3,
    val_split=0.2,
    save_path="best_eye_gaze_model.pth",
):
    dataset = EyeGazeDataset(root_dir, transform)
    val_size = max(1, int(val_split * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"[INFO] Train: {train_size}, Val: {val_size}")

    kwargs = dict(shuffle=True)
    if device.type == "cuda": kwargs["pin_memory"] = True
    train_loader = DataLoader(train_ds, batch_size=batch_size, **kwargs)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type=="cuda"))

    model = EyeGazeCNN().to(device)
    criterion = nn.SmoothL1Loss(beta=0.02)     # robust to label noise
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.2)

    best_val = float("inf")
    for epoch in range(1, num_epochs+1):
        model.train()
        run = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            run += loss.item()
        avg_tr = run / max(1, len(train_loader))

        model.eval()
        run = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                loss = criterion(preds, labels)
                run += loss.item()
        avg_val = run / max(1, len(val_loader))
        scheduler.step()

        print(f"Epoch {epoch:02d}/{num_epochs}  Train {avg_tr:.6f}  Val {avg_val:.6f}")
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Saved best -> {save_path}  (Val {best_val:.6f})")

    print("[INFO] Done.")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    print("[INFO] Best model reloaded & ready.")

if __name__ == "__main__":
    root = _sys.argv[1] if len(_sys.argv) > 1 else "eye_images"
    train(root_dir=root)
