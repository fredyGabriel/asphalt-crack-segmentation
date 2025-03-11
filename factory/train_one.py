import torch

from torch.amp import autocast
from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    scaler=None, accumulation_steps=4, show_progress=True):
    model.train()
    running_loss = 0.0

    # Zero gradients once at the beginning of accumulation cycle
    optimizer.zero_grad()

    # Create progress bar
    if show_progress:
        progress_bar = tqdm(total=len(dataloader), desc="Training",
                            leave=False, position=0)

    for i, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)

        # Use mixed precision where available
        if scaler is not None:
            with autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)
                # Scale loss by accumulation steps
                loss = criterion(outputs, masks) / accumulation_steps

            # Accumulate scaled gradients
            scaler.scale(loss).backward()

            # Only update weights after accumulation_steps
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

        # Calculate batch loss
        batch_loss = loss.item() * accumulation_steps
        running_loss += batch_loss

        # Update progress bar with current batch information
        if show_progress:
            batch_metrics = {}

            # Add instant loss to metrics
            batch_metrics["loss"] = batch_loss

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(
                {k: f"{v:.4f}" for k, v in batch_metrics.items()})

    # Close progress bar
    if show_progress:
        progress_bar.close()

    return running_loss / len(dataloader)
