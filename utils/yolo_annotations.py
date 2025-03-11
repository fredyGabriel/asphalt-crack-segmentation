import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import ndimage


# TODO Need improvement
def generar_anotaciones_yolo(carpeta_mascaras, archivo_sin_fisuras,
                             carpeta_salida):
    """
    Generates YOLO format annotations for crack detection from binary masks.

    Args:
        carpeta_mascaras (str): Path to folder with binary masks
        archivo_sin_fisuras (str): Path to .txt file with list of images
        without cracks
        carpeta_salida (str): Folder where annotation files will be saved
    """
    # Create output folder if it doesn't exist
    os.makedirs(carpeta_salida, exist_ok=True)

    # Load list of images without cracks
    imagenes_sin_fisuras = set()
    if os.path.exists(archivo_sin_fisuras):
        with open(archivo_sin_fisuras, 'r') as f:
            imagenes_sin_fisuras = set(line.strip() for line in f.readlines())

    # Valid extensions
    extensiones_validas = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # List all masks
    archivos = [f for f in os.listdir(carpeta_mascaras)
                if os.path.splitext(f.lower())[1] in extensiones_validas
                and f not in imagenes_sin_fisuras]

    print(f"Processing {len(archivos)} images with cracks...")

    for archivo in tqdm(archivos, desc="Generating YOLO annotations"):
        nombre_base = os.path.splitext(archivo)[0]
        ruta_mascara = os.path.join(carpeta_mascaras, archivo)

        try:
            # Load mask
            mascara = np.array(Image.open(ruta_mascara).convert('L'))
            altura, anchura = mascara.shape

            # Apply connected components method
            # to identify separate groups of cracks
            mascara_bin = (mascara > 127).astype(np.uint8)

            # If there are no white pixels, skip this image
            if not np.any(mascara_bin):
                continue

            # Label connected components
            etiquetado, num_componentes = ndimage.label(mascara_bin)

            # For each component, generate a bounding box
            anotaciones = []

            for componente in range(1, num_componentes + 1):
                # Get coordinates of pixels in this component
                coordenadas = np.where(etiquetado == componente)

                if len(coordenadas[0]) < 10:  # Ignore very small components
                    continue

                # Find extremes
                y_min = np.min(coordenadas[0])
                y_max = np.max(coordenadas[0])
                x_min = np.min(coordenadas[1])
                x_max = np.max(coordenadas[1])

                # Calculate center and normalized dimensions (YOLO format)
                x_centro = (x_min + x_max) / (2 * anchura)
                y_centro = (y_min + y_max) / (2 * altura)
                ancho = (x_max - x_min) / anchura
                alto = (y_max - y_min) / altura

                # Class 0 for crack
                anotaciones.append(f"0 {x_centro:.6f} {y_centro:.6f} \
{ancho:.6f} {alto:.6f}")

            # Save annotations for this image
            if anotaciones:
                ruta_salida = os.path.join(carpeta_salida, f"{nombre_base}.txt"
                                           )
                with open(ruta_salida, 'w') as f:
                    for anotacion in anotaciones:
                        f.write(f"{anotacion}\n")

        except Exception as e:
            print(f"Error processing {archivo}: {e}")

    print(f"\nProcess complete. Annotations generated in {carpeta_salida}")


if __name__ == "__main__":
    # Configuration
    carpeta_mascaras = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
doctorado/CrackDataset/luz_crack/masks/"
    archivo_sin_fisuras = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
doctorado/CrackDataset/luz_crack/sin_fisuras.txt"
    carpeta_salida = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
doctorado/CrackDataset/luz_crack/anotaciones_yolo/"

    # Generate annotations
    generar_anotaciones_yolo(carpeta_mascaras, archivo_sin_fisuras,
                             carpeta_salida)
