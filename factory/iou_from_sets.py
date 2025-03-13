import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


def calculate_iou(pred_mask, gt_mask):
    """
    Calcula el Intersection over Union (IoU) entre una máscara de predicción y
    ground truth.

    Args:
        pred_mask (numpy.ndarray or torch.Tensor): Máscara de predicción
            binaria
        gt_mask (numpy.ndarray or torch.Tensor): Máscara de ground truth
            binaria

    Returns:
        float: Valor IoU entre 0 y 1
    """
    # Convert to PyTorch tensors if they're numpy arrays
    if isinstance(pred_mask, np.ndarray):
        pred_mask = torch.from_numpy(pred_mask)
    if isinstance(gt_mask, np.ndarray):
        gt_mask = torch.from_numpy(gt_mask)

    # Make sure tensors are on CPU
    pred_mask = pred_mask.cpu()
    gt_mask = gt_mask.cpu()

    # Binarizar las máscaras si no lo están ya (umbral a 127)
    pred_binary = pred_mask > 127
    gt_binary = gt_mask > 127

    # Calcular intersección y unión usando PyTorch
    intersection = torch.logical_and(pred_binary, gt_binary).sum().item()
    union = torch.logical_or(pred_binary, gt_binary).sum().item()

    # Evitar división por cero
    if union == 0:
        # Si ambas máscaras están vacías, IoU = 1.0
        if pred_binary.sum().item() == 0 and gt_binary.sum().item() == 0:
            return 1.0
        return 0.0

    # Calcular IoU
    iou = intersection / union
    return iou


def evaluate_predictions(predictions_dir, targets_dir, output_dir=None,
                         visualize=False):
    """
    Evalúa las predicciones calculando IoU para cada imagen.

    Args:
        predictions_dir (str): Directorio con imágenes de predicción
        targets_dir (str): Directorio con imágenes de ground truth
        output_dir (str): Directorio para guardar visualizaciones y resultados
        visualize (bool): Si es True, genera visualizaciones de comparación

    Returns:
        tuple: (iou_promedio, resultados_por_imagen)
    """
    # Crear directorio de salida si no existe
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        if visualize:
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)

    # Obtener lista de imágenes en ambos directorios
    pred_files = {f.stem: f for f in Path(predictions_dir).glob("*.png")}
    target_files = {f.stem: f for f in Path(targets_dir).glob("*.png")}

    # Encontrar imágenes comunes
    common_names = set(pred_files.keys()).intersection(set(target_files.keys())
                                                       )

    if not common_names:
        print("No se encontraron imágenes con el mismo nombre en ambos \
directorios.")
        return 0.0, {}

    print(f"Procesando {len(common_names)} imágenes...")

    # Calcular IoU para cada imagen
    results = {}
    for name in tqdm(common_names):
        pred_path = pred_files[name]
        target_path = target_files[name]

        # Cargar imágenes y convertir a tensores
        pred_img = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)

        # Verificar que ambas imágenes se cargaron correctamente
        if pred_img is None or target_img is None:
            print(f"Error al cargar imágenes para {name}")
            continue

        # Convertir a tensores de PyTorch
        pred_tensor = torch.from_numpy(pred_img)
        target_tensor = torch.from_numpy(target_img)

        # Redimensionar predicción al tamaño del target si son diferentes
        if pred_img.shape != target_img.shape:
            # Resize using OpenCV and then convert to tensor
            pred_img = cv2.resize(pred_img, (target_img.shape[1],
                                             target_img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            pred_tensor = torch.from_numpy(pred_img)

        # Normalizar polaridad: fisuras en blanco (255)
        # Este enfoque asume que las fisuras son la minoría de píxeles
        if torch.mean(pred_tensor).item() > 127:
            pred_tensor = 255 - pred_tensor

        if torch.mean(target_tensor).item() > 127:
            target_tensor = 255 - target_tensor

        # Calcular IoU usando tensores
        iou = calculate_iou(pred_tensor, target_tensor)
        results[name] = iou

        # Generar visualización
        if visualize and output_dir is not None:
            # Para visualización, convertimos a NumPy arrays
            pred_img = pred_tensor.numpy()
            target_img = target_tensor.numpy()

            plt.figure(figsize=(18, 6))

            # Predicción
            plt.subplot(131)
            plt.imshow(pred_img, cmap='gray')
            plt.title('Predicción')
            plt.axis('off')

            # Ground truth
            plt.subplot(132)
            plt.imshow(target_img, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')

            # Comparación
            plt.subplot(133)
            comparison = np.zeros(
                (target_img.shape[0], target_img.shape[1], 3), dtype=np.uint8)
            # Verde: True positives (en ambas imágenes)
            comparison[np.logical_and(pred_img > 127, target_img > 127)] =\
                [0, 255, 0]
            # Rojo: False positives (solo en predicción)
            comparison[np.logical_and(pred_img > 127, target_img <= 127)] =\
                [255, 0, 0]
            # Azul: False negatives (solo en target)
            comparison[np.logical_and(pred_img <= 127, target_img > 127)] =\
                [0, 0, 255]

            plt.imshow(comparison)
            plt.title(f'IoU: {iou:.4f}')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"{name}_comparison.png"),
                        dpi=150)
            plt.close()

    # Calcular promedio usando PyTorch
    if results:
        iou_values = torch.tensor(list(results.values()))
        average_iou = torch.mean(iou_values).item()
    else:
        average_iou = 0.0

    if output_dir is not None:
        # Guardar resultados en CSV
        df = pd.DataFrame(list(results.items()), columns=['image', 'iou'])
        df['iou'] = df['iou'].round(4)
        df.to_csv(os.path.join(output_dir, "iou_results.csv"), index=False)

        # Guardar resumen
        with open(os.path.join(output_dir, "summary.txt"), "w") as f:
            f.write(f"Total de imágenes evaluadas: {len(results)}\n")
            f.write(f"IoU promedio: {average_iou:.4f}\n")
            f.write(f"IoU mínimo: {min(results.values()):.4f}\n")
            f.write(f"IoU máximo: {max(results.values()):.4f}\n")

    print(f"IoU promedio: {average_iou:.4f}")
    return average_iou, results


if __name__ == "__main__":
    # Directorios de entrada y salida
    predictions_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
doctorado/CrackDataset/luz_crack/5b-otsu_threshold/"
    targets_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/masks/"
    output_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/evaluation_results/"

    # Evaluar predicciones
    avg_iou, results = evaluate_predictions(
        predictions_dir=predictions_dir,
        targets_dir=targets_dir,
        output_dir=output_dir,
        visualize=True  # Generar visualizaciones de comparación
    )

    # Mostrar imágenes con mejor y peor IoU
    if results:
        best_image = max(results.items(), key=lambda x: x[1])[0]
        worst_image = min(results.items(), key=lambda x: x[1])[0]

        print(f"\nMejor IoU: {best_image} con {results[best_image]:.4f}")
        print(f"Peor IoU: {worst_image} con {results[worst_image]:.4f}")
