
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import src.config as config
from src.ai.value_model import ValueNetwork
import os
import argparse

def train_value_net(data_file="value_data.npz", epochs=200, batch_size=2048):
    if not os.path.exists(data_file):
        print(f"File {data_file} not found!")
        return
        
    print("Loading value data...")
    data = np.load(data_file)
    states = data['states']
    values = data['values']
    
    print(f"Loaded {len(states)} samples.")
    
    X = torch.FloatTensor(states)
    y = torch.FloatTensor(values).unsqueeze(1) # [Batch, 1]
    
    # Large batch size for speed on CPU/Simple GPU
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = ValueNetwork(hidden_size=1024)
    criterion = nn.MSELoss() 
    
    # Higher LR for larger batch size
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    
    print(f"Starting Fast Training (Batch {batch_size}, LR 0.002)...")
    
    if not os.path.exists("checkpoints_value"):
        os.makedirs("checkpoints_value")
    
    best_loss = float('inf')
    patience = 20 # Reduced patience
    patience_counter = 0
        
    for epoch in range(epochs):
        model.train() # Enable Dropout
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        rmse = np.sqrt(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints_value/value_net_final.pth")
        else:
            patience_counter += 1
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss (MSE) {avg_loss:.2f}, RMSE (Pts) {rmse:.2f} (Best: {np.sqrt(best_loss):.2f})")
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    # Load best model for saving logic
    # Already saved best_model to final.pth during loop
    print("Training complete. Best model saved to checkpoints_value/value_net_final.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    train_value_net(epochs=args.epochs)
