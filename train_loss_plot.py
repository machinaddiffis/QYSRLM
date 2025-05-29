import pickle
import matplotlib.pyplot as plt

def plot_losses(loss_path):
    """
    Load loss history from a pickle file and plot train/val losses over epochs.
    """
    with open(loss_path, 'rb') as f:
        history = pickle.load(f)
    train_losses = history['train']
    val_losses = history['val']
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

plot_losses("losses_MLP.pkl")