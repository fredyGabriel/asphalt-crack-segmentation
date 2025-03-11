import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


def calculate_f1_score(pred_mask, gt_mask):
    """
    Calcula el F1-Score entre una máscara de predicción y ground truth.
    F1-Score es la media armónica de precisión y recall.

    Args:
        pred_mask (numpy.ndarray): Máscara de predicción binaria
        gt_mask (numpy.ndarray): Máscara de ground truth binaria

    Returns:
        tuple: (f1_score, precision, recall) - valores entre 0 y 1
    """
    # Binarizar las máscaras si no lo están ya (umbral a 127)
    pred_binary = pred_mask > 127
    gt_binary = gt_mask > 127

    # Calcular métricas básicas
    true_positive = np.logical_and(pred_binary, gt_binary).sum()
    false_positive = np.logical_and(pred_binary, np.logical_not(gt_binary)
                                    ).sum()
    false_negative = np.logical_and(np.logical_not(pred_binary), gt_binary
                                    ).sum()

    # Calcular precisión y recall
    precision = true_positive / (true_positive + false_positive) if (
        true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (
        true_positive + false_negative) > 0 else 0.0

    # Calcular F1-Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (
        precision + recall) > 0 else 0.0

    return f1_score, precision, recall


def evaluate_predictions_f1(predictions_dir, targets_dir, output_dir=None,
                            visualize=False):
    """
    Evalúa las predicciones calculando F1-Score, precisión y recall para cada
    imagen.

    Args:
        predictions_dir (str): Directorio con imágenes de predicción
        targets_dir (str): Directorio con imágenes de ground truth
        output_dir (str): Directorio para guardar visualizaciones y resultados
        visualize (bool): Si es True, genera visualizaciones de comparación

    Returns:
        tuple: (f1_promedio, resultados_por_imagen)
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

    # Calcular métricas para cada imagen
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
            pred_img = cv2.resize(pred_img,
                                  (target_img.shape[1], target_img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        # Normalizar polaridad: fisuras en blanco (255)
        # Este enfoque asume que las fisuras son la minoría de píxeles
        if np.mean(pred_img) > 127:
            pred_img = 255 - pred_img

        if np.mean(target_img) > 127:
            target_img = 255 - target_img

        # Calcular métricas
        f1, precision, recall = calculate_f1_score(pred_img, target_img)
        iou = calculate_iou(pred_img, target_img)

        # Guardar resultados
        results[name] = {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'iou': iou
        }

        # Generar visualización
        if visualize and output_dir is not None:
            plt.figure(figsize=(15, 10))

            # Predicción
            plt.subplot(231)
            plt.imshow(pred_img, cmap='gray')
            plt.title('Predicción')
            plt.axis('off')

            # Ground truth
            plt.subplot(232)
            plt.imshow(target_img, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')

            # Comparación
            plt.subplot(233)
            comparison = np.zeros((target_img.shape[0],
                                   target_img.shape[1], 3), dtype=np.uint8)
            # Verde: True positives (en ambas imágenes)
            comparison[np.logical_and(pred_img > 127, target_img > 127)
                       ] = [0, 255, 0]
            # Rojo: False positives (solo en predicción)
            comparison[np.logical_and(pred_img > 127, target_img <= 127)
                       ] = [255, 0, 0]
            # Azul: False negatives (solo en target)
            comparison[np.logical_and(pred_img <= 127, target_img > 127)
                       ] = [0, 0, 255]

            plt.imshow(comparison)
            plt.title(f'F1: {f1:.4f} | IoU: {iou:.4f}')
            plt.axis('off')

            # Métricas en un gráfico de barras
            plt.subplot(212)
            metrics = ['F1-Score', 'Precisión', 'Recall', 'IoU']
            values = [results[name]['f1_score'], results[name]['precision'],
                      results[name]['recall'], results[name]['iou']]

            bars = plt.bar(metrics, values, color=['purple', 'green', 'blue',
                                                   'orange'])
            plt.ylim([0, 1])
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Añadir valores sobre las barras
            for bar, val in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                         f'{val:.3f}', ha='center', fontweight='bold')

            plt.title(f'Métricas de evaluación para {name}')

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"{name}_evaluation.png"),
                        dpi=150)
            plt.close()

    # Calcular promedios y guardar resultados
    f1_scores = [r['f1_score'] for r in results.values()]
    precisions = [r['precision'] for r in results.values()]
    recalls = [r['recall'] for r in results.values()]
    ious = [r['iou'] for r in results.values()]

    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_iou = np.mean(ious)

    if output_dir is not None:
        # Guardar resultados en CSV
        rows = []
        for name, metrics in results.items():
            rows.append({
                'image': name,
                'f1_score': round(metrics['f1_score'], 4),
                'precision': round(metrics['precision'], 4),
                'recall': round(metrics['recall'], 4),
                'iou': round(metrics['iou'], 4)
            })

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, "metrics_results.csv"), index=False)

        # Guardar resumen
        with open(os.path.join(output_dir, "summary.txt"), "w") as f:
            f.write(f"Total de imágenes evaluadas: {len(results)}\n\n")
            f.write(f"F1-Score promedio: {avg_f1:.4f}\n")
            f.write(f"Precisión promedio: {avg_precision:.4f}\n")
            f.write(f"Recall promedio: {avg_recall:.4f}\n")
            f.write(f"IoU promedio: {avg_iou:.4f}\n\n")

            # Imágenes con mejor y peor F1-Score
            best_f1_image = max(results.items(),
                                key=lambda x: x[1]['f1_score'])[0]
            worst_f1_image = min(results.items(),
                                 key=lambda x: x[1]['f1_score'])[0]

            f.write(f"Mejor F1-Score: {best_f1_image} con \
{results[best_f1_image]['f1_score']:.4f}\n")
            f.write(f"Peor F1-Score: {worst_f1_image} con \
{results[worst_f1_image]['f1_score']:.4f}\n")

        # Generar gráfico comparativo de todas las métricas
        plt.figure(figsize=(10, 6))
        metrics_avg = {
            'F1-Score': avg_f1,
            'Precisión': avg_precision,
            'Recall': avg_recall,
            'IoU': avg_iou
        }

        bars = plt.bar(metrics_avg.keys(), metrics_avg.values(),
                       color=['purple', 'green', 'blue', 'orange'])

        plt.ylim([0, 1])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.title('Métricas Promedio para Todo el Conjunto de Datos')

        # Añadir valores sobre las barras
        for bar, val in zip(bars, metrics_avg.values()):
            plt.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                     f'{val:.4f}', ha='center', fontweight='bold')

        plt.savefig(os.path.join(output_dir, "average_metrics.png"), dpi=150)
        plt.close()

    print(f"F1-Score promedio: {avg_f1:.4f}")
    print(f"Precisión promedio: {avg_precision:.4f}")
    print(f"Recall promedio: {avg_recall:.4f}")
    return avg_f1, results


# Función IoU del archivo original para compatibilidad
def calculate_iou(pred_mask, gt_mask):
    """
    Calcula el Intersection over Union (IoU) entre una máscara de predicción y
    ground truth.
    """
    # Binarizar las máscaras
    pred_binary = pred_mask > 127
    gt_binary = gt_mask > 127

    # Calcular intersección y unión
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    # Evitar división por cero
    if union == 0:
        if np.sum(pred_binary) == 0 and np.sum(gt_binary) == 0:
            return 1.0
        return 0.0

    # Calcular IoU
    iou = intersection / union
    return iou


if __name__ == "__main__":
    # Directorios de entrada y salida
    predictions_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
doctorado/CrackDataset/luz_crack/5c-gauss_adaptive_threshold/"
    targets_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/masks/"
    output_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/evaluation_f1_gauss/"

    # Evaluar predicciones
    avg_f1, results = evaluate_predictions_f1(
        predictions_dir=predictions_dir,
        targets_dir=targets_dir,
        output_dir=output_dir,
        visualize=True  # Generar visualizaciones de comparación
    )

    # Mostrar imágenes con mejor y peor F1-Score
    if results:
        best_image = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
        worst_image = min(results.items(), key=lambda x: x[1]['f1_score'])[0]

        print(f"\nMejor F1-Score: {best_image} con \
{results[best_image]['f1_score']:.4f}")
        print(f"Peor F1-Score: {worst_image} con \
{results[worst_image]['f1_score']:.4f}")
