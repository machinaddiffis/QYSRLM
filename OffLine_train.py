import numpy as np
import pickle

# file_path="./data/Baseline_58000_0.pkl"
# with open(file_path, "rb") as file:
#     all_data = pickle.load(file)
# print(len(all_data))
# print(all_data[0][0][0])



import os
import pickle
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utlis import PKLDataset,train_one_epoch,validate
from models import FCNN,ResidualMLP



def main(args):
    # Prepare file lists
    data_dir = args.data_dir
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')])
    train_files = all_files[:200]
    val_files = all_files[-5:]

    # Create datasets and loaders
    train_dataset = PKLDataset(train_files)
    val_dataset = PKLDataset(val_files)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model, loss, optimizer
    input_dim = 8 * 19
    output_dim = 8 * 17
    # model = FCNN(input_dim, args.hidden_dim, output_dim).to(device)
    model=ResidualMLP(input_dim, args.hidden_dim, output_dim, num_blocks=3).to(device)
    criterion = nn.MSELoss()
    # criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss,train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,eval_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.6f}- train acc: {train_acc:.6f} - Val Loss: {val_loss:.6f},- eval acc: {eval_acc:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)

        history = {'train': train_losses, 'val': val_losses}
        with open(args.loss_path, 'wb') as f:
            pickle.dump(history, f)
    print(f"Saved loss history to {args.loss_path}")

    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a simple FCNN on PKL data')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing .pkl files')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden layer dimension')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--save_path', type=str, default='best_model_MLP.pth', help='Where to save the best model')
    parser.add_argument('--loss_path', type=str, default='losses_MLP.pkl', help='Where to save loss history')
    args = parser.parse_args()
    main(args)

