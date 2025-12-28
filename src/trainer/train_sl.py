import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import json

# Fix imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.ai.model import YahtzeeNetwork
import src.config as config

def train_sl(data_file="expert_data.npz", epochs=50, batch_size=64):
    print("Loading data...")
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
        return

    data = np.load(data_file)
    states = torch.FloatTensor(data['states'])
    actions = torch.LongTensor(data['actions'])
    
    print(f"Loaded {len(states)} samples.")
    
    # Dataset
    dataset = torch.utils.data.TensorDataset(states, actions)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = YahtzeeNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    if not os.path.exists("checkpoints_sl"):
        os.makedirs("checkpoints_sl")
        
    history = []
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_states, batch_actions in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += batch_actions.size(0)
            correct += (predicted == batch_actions).sum().item()
            
        avg_loss = total_loss / len(dataloader)
        acc = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{epochs}: Loss {avg_loss:.4f}, Accuracy {acc:.2f}%")
        
        history.append({"epoch": epoch, "loss": avg_loss, "accuracy": acc})
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints_sl/sl_model_ep{epoch+1}.pth")
            
    # Save Final
    final_path = "checkpoints_sl/sl_model_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Model saved to {final_path}")
    
    # Save Log
    with open("checkpoints_sl/training_log.json", "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    train_sl(epochs=args.epochs)
