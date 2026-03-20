import matplotlib.pyplot as plt  # pyre-ignore[21]
import torch  # pyre-ignore[21]

def save_plots(train_losses, val_accs, filename="results.png"):
    """Plots training loss and accuracy rates."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Species Acc')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.savefig(filename)
    plt.show()

def load_checkpoint(model, path):
    """Loads saved model weights."""
    model.load_state_dict(torch.load(path))
    model.eval()
    return model