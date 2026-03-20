import torch  # pyre-ignore[21]
from src.data_loader import data_loader  # pyre-ignore[21]
from src.model import MultiHeadBirdModel  # pyre-ignore[21]
from src.train import train_model  # pyre-ignore[21]
from src.utils import save_plots  # pyre-ignore[21]

# --- Enter the paths to the data here ---
TRAIN_CSV = "data/nabirds/train.csv"   # training CSV file path
VAL_CSV   = "data/nabirds/val.csv"     # validation CSV file path
IMG_DIR   = "data/nabirds/images"      # images root folder path

# --- Training configuration (speed/stability) ---
BATCH_SIZE = 128
IMG_SIZE = 192
EPOCHS = 30
LR = 1e-4

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Create DataLoader
    train_loader, val_loader = data_loader(
        TRAIN_CSV,
        VAL_CSV,
        IMG_DIR,
        batch_size=BATCH_SIZE,
        num_workers=None,
        img_size=IMG_SIZE,
    )

    # num_species=555: class IDs are remapped 0-554 (full NABirds range).
    # Only 228 of these classes are present on disk; the others simply never appear
    # in training data, so their output nodes are unused but cause no errors.
    model = MultiHeadBirdModel(num_species=555).to(device)
    
    # Start training
    print("Training is starting...")
    history = train_model(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LR)
    
    # Save plots
    save_plots(history['train_loss'], history['val_accs'])
    
    # Save final model
    torch.save(model.state_dict(), "final_bird_model.pth")
    print("Training is completed and plots are generated!")

if __name__ == "__main__":
    main()