import torch
from factory.metrics import iou, precision, recall, f1_score
import yaml
from tqdm import tqdm


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def validate(model, dataloader, criterion, device, show_progress=True):
    """Validation function"""
    model.eval()
    running_loss = 0.0

    # Create progress bar
    if show_progress:
        progress_bar = tqdm(total=len(dataloader), desc="Validating",
                            leave=False, position=0)

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            if show_progress:
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    if show_progress:
        progress_bar.close()

    return running_loss / len(dataloader)


def calculate_metrics(model, dataloader, device, config, threshold=0.3,
                      show_progress=True):
    model.eval()

    # Lists to store all predictions and targets
    all_predictions = []
    all_masks = []

    if show_progress:
        progress_bar = tqdm(total=len(dataloader),
                            desc="Collecting data for metrics", leave=False,
                            position=0)

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predictions = (probs > threshold).float()

            # Store predictions and masks
            all_predictions.append(predictions.cpu())
            all_masks.append(masks.cpu())

            if show_progress:
                progress_bar.update(1)

    if show_progress:
        progress_bar.close()

    # Concatenate all predictions and masks
    all_predictions = torch.cat(all_predictions).to(device)
    all_masks = torch.cat(all_masks).to(device)

    # Get list of metrics from configuration
    configured_metrics = config.get('evaluation', {}).get('metrics', [])
    configured_metrics = [metric.lower() for metric in configured_metrics]

    # Ensure iou is always included
    if 'iou' not in configured_metrics:
        configured_metrics.append('iou')

    # Mapping metric names to functions - standardized to lowercase
    metric_functions = {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1_score
    }

    # Initialize dictionary with only configured metrics
    metrics = {}
    for metric_name in configured_metrics:
        if metric_name in metric_functions:
            metrics[metric_name] = metric_functions[metric_name](
                all_predictions, all_masks).item()

    return metrics


if __name__ == "__main__":
    pass
