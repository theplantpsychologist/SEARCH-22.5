import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import sqlite3
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Import the dimension from your build script
from database.build_database import DIMENSION

class AncestryDataset(Dataset):
    def __init__(self, db_path='database/storage/database_3.db', num_gens=5):
        self.num_gens = num_gens
        print(f"Connecting to {db_path} and building lineage map...")
        conn = sqlite3.connect(db_path)
        
        # Load all IDs and Parent IDs into RAM for fast traversal
        df = pd.read_sql_query("SELECT id, parent_id, embedding FROM states", conn)
        conn.close()

        # Map ID -> Embedding and ID -> Parent_ID
        self.id_to_embed = {row['id']: np.frombuffer(row['embedding'], dtype=np.float32) for _, row in df.iterrows()}
        self.id_to_parent = {row['id']: row['parent_id'] for _, row in df.iterrows()}
        self.ids = list(self.id_to_embed.keys())
        
        # Pre-calculate labels
        self.data = []
        for child_id in tqdm(self.ids, desc="Mapping Ancestry"):
            child_embed = self.id_to_embed[child_id]
            ancestors = []
            curr_id = child_id
            
            for _ in range(num_gens):
                parent_id = self.id_to_parent.get(curr_id)
                if parent_id is not None and parent_id in self.id_to_embed:
                    ancestors.append(self.id_to_embed[parent_id])
                    curr_id = parent_id
                else:
                    ancestors.append(np.zeros(DIMENSION, dtype=np.float32))
            
            mask = [1.0 if not np.all(a == 0) else 0.0 for a in ancestors]
            self.data.append((child_embed, np.array(ancestors), np.array(mask)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        child, ancestors, mask = self.data[idx]
        return torch.tensor(child), torch.tensor(ancestors), torch.tensor(mask)

class AncestryEnsemble(nn.Module):
    def __init__(self, input_dim=DIMENSION, num_heads=5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LeakyReLU()
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Linear(256, input_dim)
            ) for _ in range(num_heads)
        ])

    def forward(self, x):
        features = self.backbone(x)
        return torch.stack([head(features) for head in self.heads], dim=1)

def train_ancestry_model(train_time_limit=3600, batch_size=1024, num_gens=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    full_dataset = AncestryDataset(num_gens=num_gens)
    test_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - test_size
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

    print("Moving data to RAM")
    # 1. Convert lists of arrays to single large NumPy blocks first (MUCH faster)
    # This assumes d[0], d[1], d[2] are the child, ancestors, and mask
    raw_x = np.array([d[0] for d in train_ds])
    raw_y = np.array([d[1] for d in train_ds])
    raw_m = np.array([d[2] for d in train_ds])

    # 2. Move to Torch and send to GPU in one shot
    # No more UserWarnings!
    train_x = torch.from_numpy(raw_x).to(device)
    train_y = torch.from_numpy(raw_y).to(device)
    train_m = torch.from_numpy(raw_m).to(device)

    print(f"Memory Check: {train_y.element_size() * train_y.nelement() / 1e6:.2f} MB allocated for Ancestors.")
    num_train = train_x.size(0)
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cuda'))
    train_loader = DataLoader(
        train_ds, 
        batch_size=4096,           # Increased batch size to give GPU more to chew on
        shuffle=True, 
        num_workers=4,            # Adjust based on your CPU (start with 4)
        pin_memory=True,          # Essential for NVIDIA GPUs
        persistent_workers=True   # Keeps the workers alive between epochs
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, pin_memory=(device.type == 'cuda'))

    model = AncestryEnsemble(num_heads=num_gens).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss(reduction='none') 

    checkpoint_path = "ancestry_ensemble_best.pth"

    if os.path.exists(checkpoint_path):
        print(f"Found existing model at {checkpoint_path}. Loading weights...")
        # map_location ensures it loads correctly even if you move between CPU/GPU
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Resuming training from previous best state.")
    else:
        print("No checkpoint found. Starting training from scratch.")
    
    # Track loss per generation
    history = {'train': [[] for _ in range(num_gens)], 'test': [[] for _ in range(num_gens)]}
    start_time = time.time()
    epoch = 0
    best_val_loss = float('inf')
    while (time.time() - start_time) < train_time_limit:
        model.train()
        epoch_train_losses = np.zeros(num_gens)
        counts_train = np.zeros(num_gens)
        indices = torch.randperm(num_train, device=device)
        
        for batch_idx in tqdm(range(0, num_train, batch_size), desc=f"Epoch {epoch} [Train]"):
            child = train_x[indices[batch_idx:batch_idx + batch_size]]
            targets = train_y[indices[batch_idx:batch_idx + batch_size]]
            mask = train_m[indices[batch_idx:batch_idx + batch_size]]

            # child, targets, mask = child.to(device), targets.to(device), mask.to(device)
            optimizer.zero_grad()

            outputs = model(child)

            loss_raw = criterion(outputs, targets).mean(dim=-1) # [Batch, num_gens]
            loss = (loss_raw * mask).sum() / (mask.sum() + 1e-6)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_losses += loss_raw.sum(dim=0).detach().cpu().numpy()
            counts_train += mask.sum(dim=0).detach().cpu().numpy()

            if (time.time() - start_time) > train_time_limit: break

        # Record Train Loss
        for i in range(num_gens):
            history['train'][i].append(epoch_train_losses[i] / counts_train[i] if counts_train[i] > 0 else 0)

        # Validation
        model.eval()
        epoch_test_losses = np.zeros(num_gens)
        counts_test = np.zeros(num_gens)
        with torch.no_grad():
            for child, targets, mask in test_loader:
                child, targets, mask = child.to(device), targets.to(device), mask.to(device)
                outputs = model(child)
                loss_raw = criterion(outputs, targets).mean(dim=-1)
                masked_loss = loss_raw * mask
                epoch_test_losses += masked_loss.sum(dim=0).cpu().numpy()
                counts_test += mask.sum(dim=0).cpu().numpy()
        
        for i in range(num_gens):
            history['test'][i].append(epoch_test_losses[i] / counts_test[i] if counts_test[i] > 0 else 0)
        
        print(f"Epoch {epoch} | Gen 1 Train Loss: {history['train'][0][-1]:.6f} | Val: {history['test'][0][-1]:.6f}")
        epoch += 1
        val_loss = history['test'][0][-1]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "ancestry_ensemble_best.pth")
            print(f" ---> Saved new best model weights (Loss: {val_loss:.6f})")

    # --- PLOTTING ---
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, num_gens))
    gen_labels = ["Parent", "Grandparent", "3rd Gen", "4th Gen", "5th Gen"]
    
    for i in range(num_gens):
        plt.plot(history['train'][i], label=f"{gen_labels[i]} (Train)", color=colors[i], linestyle='-')
        plt.plot(history['test'][i], label=f"{gen_labels[i]} (Test)", color=colors[i], linestyle='--')
    
    renders_dir = "renders"
    os.makedirs(renders_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(renders_dir) if f.endswith(".png")]
    file_count = len(existing_files)

    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Masked MSE Loss')
    plt.title('Ancestry Ensemble Loss per Generation')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'renders/ancestry_per_gen_loss_{file_count}.png')
    
    torch.save(model.state_dict(), "database/storage/ancestry_ensemble.pth")
    print(f"Training complete. Final plot and model saved.")



if __name__ == "__main__":
    train_ancestry_model(train_time_limit=60 * 20)