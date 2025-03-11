import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # Para mostrar una barra de progreso


def binarizar_imagenes(carpeta_origen, carpeta_destino, umbral=0.5):
    """
    Binariza todas las imágenes en una carpeta y las guarda en otra.

    Args:
        carpeta_origen (str): Ruta de la carpeta con imágenes originales
        carpeta_destino (str): Ruta donde guardar las imágenes binarizadas
        umbral (float, opcional): Umbral para la binarización (0.0-1.0).
            Default: 0.5

    Returns:
        int: Número de imágenes procesadas
    """
    # Asegurar que la carpeta de destino existe
    os.makedirs(carpeta_destino, exist_ok=True)

    # Extensiones válidas de imagen
    extensiones_validas = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # Lista todos los archivos de imagen en la carpeta de origen
    archivos = [f for f in os.listdir(carpeta_origen)
                if os.path.splitext(f.lower())[1] in extensiones_validas]

    contador = 0

    # Procesa cada imagen con una barra de progreso
    for archivo in tqdm(archivos, desc="Binarizando imágenes"):
        ruta_completa = os.path.join(carpeta_origen, archivo)

        try:
            # Cargar imagen
            imagen = Image.open(ruta_completa)

            # Convertir a escala de grises
            imagen_gris = imagen.convert('L')

            # Binarizar imagen
            imagen_array = np.array(imagen_gris)
            imagen_bin_array = ((imagen_array / 255 > umbral) * 255
                                ).astype(np.uint8)
            imagen_bin = Image.fromarray(imagen_bin_array)

            # Guardar imagen binarizada con el mismo nombre
            ruta_destino = os.path.join(carpeta_destino, archivo)
            imagen_bin.save(ruta_destino)

            contador += 1

        except Exception as e:
            print(f"Error al procesar {archivo}: {e}")

    print(f"\nProceso completado: {contador} imágenes binarizadas")
    return contador


# Ejemplo de uso
if __name__ == "__main__":
    carpeta_origen = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
        doctorado/CrackDataset/luz_crack/masks/"
    carpeta_destino = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
        doctorado/CrackDataset/luz_crack/masks_bin/"

    # Llamada a la función
    num_procesadas = binarizar_imagenes(carpeta_origen, carpeta_destino)
    print(f"Total de imágenes procesadas: {num_procesadas}")
