import sys
import os
# Añadir el directorio raíz del proyecto al path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torchvision import transforms  # noqa: E402
from models.unet_resnet import UNetResNet  # noqa: E402
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402
import os  # noqa: E402


class Inference:
    def __init__(self, model_path, device=None):
        """
        Initialize the inference class with a trained model.

        Parameters:
            model_path: Path to the trained model weights
            device: Device to run inference on ('cuda' or 'cpu'). If None,
                will use CUDA if available.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available()
                                       else 'cpu')
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Initialize model with correct parameters
        self.model = UNetResNet(
            backbone="resnet34",
            in_channels=3,
            out_channels=1)

        # Cargar correctamente desde el checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Verificar si es un checkpoint completo o solo state_dict
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint from epoch \
{checkpoint.get('epoch', 'unknown')}")
        else:
            # Intentar cargar directamente
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {model_path}")

    def preprocess_image(self, image_path):
        """
        Preprocess an image for inference.

        Parameters:
            image_path: Path to the input image

        Returns:
            Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(self.device)
        return image

    def predict(self, image_path):
        """
        Run inference on an image.

        Parameters:
            image_path: Path to the input image

        Returns:
            Binary segmentation mask as a numpy array
        """
        original_img = Image.open(image_path)
        original_size = original_img.size  # (width, height)

        image = self.preprocess_image(image_path)
        with torch.no_grad():
            output = self.model(image)
            output = torch.sigmoid(output)
            output = output.squeeze().cpu().numpy()

        mask_resized = Image.fromarray((output > 0.5).astype(np.uint8))
        mask_resized = mask_resized.resize((original_size[0], original_size[1]
                                            ),
                                           Image.NEAREST)
        return np.array(mask_resized)

    def save_mask(self, mask, save_path):
        """
        Save a segmentation mask as an image.

        Parameters:
            mask: Binary segmentation mask
            save_path: Path to save the output mask
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mask_image = Image.fromarray(mask * 255)
        mask_image.save(save_path)
        print(f"Mask saved to {save_path}")

    def apply_postprocessing(self, mask, morph_op='close'):
        import cv2
        kernel = np.ones((3, 3), np.uint8)
        if morph_op == 'close':
            return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Añadir otras operaciones según necesidades
        return mask


def process_single_image(model_path, image_path, save_path):
    """
    Process a single image and save the segmentation mask.

    Parameters:
        model_path: Path to the trained model weights
        image_path: Path to the input image
        save_path: Path to save the output mask
    """
    inference = Inference(model_path=model_path)
    mask = inference.predict(image_path)
    inference.save_mask(mask, save_path)


def process_directory(model_path, input_dir, output_dir):
    """
    Process all images in a directory and save the segmentation masks.

    Parameters:
        model_path: Path to the trained model weights
        input_dir: Directory containing input images
        output_dir: Directory to save output masks
    """
    inference = Inference(model_path=model_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the input directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    processed_count = 0

    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            image_path = os.path.join(input_dir, filename)
            save_path = os.path.join(output_dir,
                                     filename.replace(ext, '_mask.png'))

            try:
                mask = inference.predict(image_path)
                inference.save_mask(mask, save_path)
                processed_count += 1
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Completed. Processed {processed_count} images.")


if __name__ == "__main__":
    # Example usage:
    # 1. Process a single image
    model_path = "saved_models/model4/best_iou_model.pth"
    image_path = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/images/52_1.jpg"
    save_path = "saved_models/model4/pred52_1.png"
    process_single_image(model_path, image_path, save_path)

    # 2. Process all images in a directory
    # model_path = "models/best_model.pth"
    # input_dir = "data/test"
    # output_dir = "results/masks"
    # process_directory(model_path, input_dir, output_dir)

    # Uncomment and modify one of the above examples to run
    # print("Uncomment and modify one of the example usages in the script")
