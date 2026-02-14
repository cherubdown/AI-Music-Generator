import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset import BachDataset
from src.models.lstm import BachLSTM
import argparse
from pathlib import Path
from tqdm import tqdm
import os

def train(args):
    # Device configuration
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # Test device
            torch.zeros(1).to(device)
            print(f"Using device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("Using device: cpu")
    except Exception as e:
        print(f"CUDA error: {e}")
        print("Falling back to CPU.")
        device = torch.device('cpu')
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Hyperparameters
    input_size = 128
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    num_classes = 128
    sequence_length = args.seq_length
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs
    
    # Dataset
    print("Loading dataset...")
    dataset = BachDataset(root="data/jsb_chorales", download=True, seq_length=sequence_length, transpose=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)} sequences")
    
    # Model
    model = BachLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    
    # Loss and optimizer
    # For multi-label (polyphonic), we use BCEWithLogitsLoss
    # For single-note (monophonic), we use CrossEntropyLoss
    # JSB Chorales are polyphonic (4 voices). 
    # But here we flattened to piano roll (0/1).
    # So we should use BCEWithLogitsLoss to predict presence of each note.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train
    print("Starting training...")
    total_step = len(dataloader)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
            
            for i, (images, labels) in enumerate(tepoch):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/total_step:.4f}')
        
        # Save model
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f'model_epoch_{epoch+1}.pth'))

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'model_final.pth'))
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Bach LSTM')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seq_length', type=int, default=64, help='Sequence length')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/models', help='Directory to save checkpoints')
    
    args = parser.parse_args()
    train(args)
