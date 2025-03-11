import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse


def visualizar_anotaciones_yolo(carpeta_imagenes, carpeta_anotaciones,
                                carpeta_salida=None,
                                mostrar=True, color=(255, 0, 0), grosor=2):
    """
    Visualizes YOLO annotations by drawing bounding boxes on their
    corresponding images.

    Args:
        carpeta_imagenes (str): Path to folder with original images
        carpeta_anotaciones (str): Path to folder with YOLO annotation .txt
            files
        carpeta_salida (str, optional): Folder to save visualized images with
            annotations
        mostrar (bool): Whether to display images during execution
        color (tuple): Bounding box color in (R,G,B) format
        grosor (int): Thickness of bounding box line
    """
    # Create output folder if necessary
    if carpeta_salida:
        os.makedirs(carpeta_salida, exist_ok=True)

    # Search for image and annotation pairs
    extensiones_validas = {'.jpg', '.jpeg', '.png', '.bmp'}

    # Get all annotation files
    archivos_anotaciones = [f for f in os.listdir(carpeta_anotaciones)
                            if f.endswith('.txt')]

    print(f"Found {len(archivos_anotaciones)} annotation files")

    # For each annotation file, look for the corresponding image
    for archivo_anotacion in tqdm(archivos_anotaciones,
                                  desc="Visualizing annotations"):
        nombre_base = os.path.splitext(archivo_anotacion)[0]

        # Look for corresponding image
        imagen_encontrada = False
        for ext in extensiones_validas:
            ruta_imagen = os.path.join(carpeta_imagenes, nombre_base + ext)
            if os.path.exists(ruta_imagen):
                imagen_encontrada = True
                break

        if not imagen_encontrada:
            print(f"No image found for {archivo_anotacion}")
            continue

        # Load image
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            imagen = np.array(Image.open(ruta_imagen))
            imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)

        alto_img, ancho_img = imagen.shape[:2]

        # Read annotations
        ruta_anotacion = os.path.join(carpeta_anotaciones, archivo_anotacion)
        with open(ruta_anotacion, 'r') as f:
            lineas = f.readlines()

        for linea in lineas:
            if not linea.strip():
                continue

            # Parse line: class x_center y_center width height
            partes = linea.strip().split(' ')
            if len(partes) != 5:
                continue

            clase, x_centro, y_centro, ancho, alto = partes

            # Convert normalized coordinates to pixels
            x_centro = float(x_centro) * ancho_img
            y_centro = float(y_centro) * alto_img
            ancho = float(ancho) * ancho_img
            alto = float(alto) * alto_img

            # Calculate rectangle coordinates
            x1 = int(x_centro - ancho / 2)
            y1 = int(y_centro - alto / 2)
            x2 = int(x_centro + ancho / 2)
            y2 = int(y_centro + alto / 2)

            # Draw rectangle
            cv2.rectangle(imagen, (x1, y1), (x2, y2), color, grosor)

            # Add class label
            etiqueta = f"Crack {clase}"
            tamaño_texto = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX,
                                           0.6, 1)[0]
            cv2.rectangle(imagen, (x1, y1 - tamaño_texto[1] - 10),
                          (x1 + tamaño_texto[0], y1), color, -1)
            cv2.putText(imagen, etiqueta, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show image if requested
        if mostrar:
            cv2.imshow(f"Annotations: {nombre_base}", imagen)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save image if output folder is specified
        if carpeta_salida:
            ruta_salida = os.path.join(carpeta_salida,
                                       f"{nombre_base}_annotated.jpg")
            cv2.imwrite(ruta_salida, imagen)

    print("Visualization completed")


if __name__ == "__main__":
    # Configuration using argparse for command-line usage
    parser = argparse.ArgumentParser(description="Visualize YOLO annotations")
    parser.add_argument("--imagenes", default="C:/Users/fgrv/OneDrive/\
                        Documentos/PythonProjects/doctorado/CrackDataset/\
                        luz_crack/images/",
                        help="Folder with original images")
    parser.add_argument("--anotaciones", default="C:/Users/fgrv/OneDrive/\
                        Documentos/PythonProjects/doctorado/CrackDataset/\
                        luz_crack/anotaciones_yolo/",
                        help="Folder with YOLO annotations (.txt)")
    parser.add_argument("--salida", default="C:/Users/fgrv/OneDrive/\
                        Documentos/PythonProjects/doctorado/CrackDataset/\
                        luz_crack/visualizaciones/",
                        help="Folder to save images with annotations")
    parser.add_argument("--mostrar", action="store_true",
                        help="Show images during execution")

    args = parser.parse_args()

    # Call visualization function
    visualizar_anotaciones_yolo(
        args.imagenes,
        args.anotaciones,
        args.salida,
        args.mostrar
    )
