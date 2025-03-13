import yaml
import timm
import torch


def load_pretrained_model():
    """
    Load a pretrained model from timm.

    Returns:
        torch.nn.Module: Pretrained model
    """
    # Modelo recomendado para segmentación de fisuras
    model = timm.create_model('swinv2_small_window16_256.ms_in1k',
                              pretrained=True)

    # Guardar pesos para tu implementación
    torch.save(model.state_dict(),
               'saved_models/swinv2_small_window16_256_in1k.pth')


def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    # Example usage
    config_path = "configs/default.yaml"
    config = load_config(config_path)

    # Create a model based on config
    model = load_pretrained_model()
    print(f"Created model: {type(model).__name__}")
