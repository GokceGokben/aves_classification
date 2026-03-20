import os
import torch  # pyre-ignore[21]
import torch.nn as nn  # pyre-ignore[21]
import torch.optim as optim  # pyre-ignore[21]
from tqdm import tqdm  # pyre-ignore[21]


def evaluate(model, loader, device, amp_enabled=False):
    """Runs the model on loader and returns accuracy (%)."""
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            targets = labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(images)

            _, preds = torch.max(outputs, 1)

            total   += targets.size(0)
            correct += (preds == targets).sum().item()  # pyre-ignore

    return 100 * correct / total


def train_model(model, train_loader, val_loader, device, epochs=30, lr=1e-4):
    """
    Trains model for `epochs` epochs and returns a history dict:
        {
            'train_loss': [float, ...],
            'val_accs':   {'species': [...], 'gender': [...], 'age': [...]}
        }
    """
    amp_enabled = (device.type == 'cuda')

    optimizer         = optim.AdamW(model.parameters(), lr=lr)
    scheduler         = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion         = nn.CrossEntropyLoss()
    scaler            = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(root_dir, 'training_checkpoint.pth')

    history = {
        'train_loss': [],
        'val_accs':   []
    }
    best_acc = 0.0
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint.get('scaler_state_dict', scaler.state_dict()))
        history = checkpoint.get('history', history)
        best_acc = checkpoint.get('best_acc', 0.0)
        start_epoch = checkpoint.get('epoch', -1) + 1
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch + 1}/{epochs}.")

    print(f"Training is starting on {device} (AMP={'ON' if amp_enabled else 'OFF'})...")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images  = images.to(device, non_blocking=True)
            targets = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{running_loss / (pbar.n + 1):.4f}"})

        # --- Epoch summary ---
        avg_loss = running_loss / len(train_loader)
        acc = evaluate(model, val_loader, device, amp_enabled=amp_enabled)

        history['train_loss'].append(avg_loss)  # pyre-ignore
        history['val_accs'].append(acc)  # pyre-ignore

        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
        print(f"  Val Acc: {acc:.2f}%")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_acc': best_acc,
            'history': history,
        }, checkpoint_path)
        print(f"  --> Checkpoint saved ({checkpoint_path})")

        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(root_dir, 'best_bird_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  --> New best model saved! ({save_path})")

    return history