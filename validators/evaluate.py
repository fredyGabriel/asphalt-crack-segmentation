import torch
from factory.metrics import iou, precision, recall, f1_score
from tqdm import tqdm


def validate(model, dataloader, criterion, device, metric_fns=None,
             threshold=0.5, show_progress=True):
    """
    Validation function that calculates loss and specified metrics in a single
    pass.

    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to run on
        metric_fns: List of metric functions to calculate
        threshold: Threshold for binary predictions
        show_progress: Whether to show progress bar

    Returns:
        (val_loss, metrics_dict)
    """
    model.eval()
    running_loss = 0.0

    all_predictions = []
    all_masks = []

    if show_progress:
        progress_bar = tqdm(total=len(dataloader), desc="Validating",
                            leave=False, position=0)

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            all_predictions.append(preds.cpu())
            all_masks.append(masks.cpu())

            if show_progress:
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    if show_progress:
        progress_bar.close()

    val_loss = running_loss / len(dataloader)

    metrics = {}

    if metric_fns:
        all_predictions = torch.cat(all_predictions).to(device)
        all_masks = torch.cat(all_masks).to(device)

        for metric_name, metric_fn in metric_fns.items():
            metrics[metric_name] = metric_fn(all_predictions, all_masks).item()

    return val_loss, metrics


def calculate_metrics(model, dataloader, device, config, threshold=0.5,
                      show_progress=True):
    """
    Calculates specified metrics using the validate function.

    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        device: Device to run on
        config: Configuration dictionary
        threshold: Threshold for binary predictions
        show_progress: Whether to show progress bar

    Returns:
        dict: Dictionary of calculated metrics
    """
    # Get list of metrics from configuration
    configured_metrics = ['iou', 'precision', 'recall', 'f1']
    if config:
        configured_metrics = config.get(
            'evaluation', {}).get('metrics', configured_metrics)
        configured_metrics = [metric.lower() for metric in
                              configured_metrics]

    # Mapping metric names to functions
    metric_fns = {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1_score
    }

    # Filter metric functions based on configuration
    selected_metric_fns = {
        metric_name: metric_fns[metric_name]
        for metric_name in configured_metrics if metric_name in metric_fns
    }

    val_loss, metrics = validate(model, dataloader, None, device,
                                 metric_fns=selected_metric_fns,
                                 threshold=threshold,
                                 show_progress=show_progress)
    return metrics


if __name__ == "__main__":
    pass
