import os
import cv2
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
        pred_mask (numpy.ndarray): Máscara de predicción binaria
        gt_mask (numpy.ndarray): Máscara de ground truth binaria

    Returns:
        float: Valor IoU entre 0 y 1
    """
    # Binarizar las máscaras si no lo están ya (umbral a 127)
    pred_binary = pred_mask > 127
    gt_binary = gt_mask > 127

    # Calcular intersección y unión
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    # Evitar división por cero
    if union == 0:
        # Si ambas máscaras están vacías, IoU = 1.0
        if np.sum(pred_binary) == 0 and np.sum(gt_binary) == 0:
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

        # Cargar imágenes
        pred_img = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)

        # Verificar que ambas imágenes se cargaron correctamente
        if pred_img is None or target_img is None:
            print(f"Error al cargar imágenes para {name}")
            continue

        # Redimensionar predicción al tamaño del target si son diferentes
        if pred_img.shape != target_img.shape:
            pred_img = cv2.resize(pred_img, (target_img.shape[1],
                                             target_img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        # Normalizar polaridad: fisuras en blanco (255)
        # Este enfoque asume que las fisuras son la minoría de píxeles
        if np.mean(pred_img) > 127:
            pred_img = 255 - pred_img

        if np.mean(target_img) > 127:
            target_img = 255 - target_img

        # Calcular IoU
        iou = calculate_iou(pred_img, target_img)
        results[name] = iou

        # Generar visualización
        if visualize and output_dir is not None:
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

    # Calcular promedio y guardar resultados
    average_iou = np.mean(list(results.values()))

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
