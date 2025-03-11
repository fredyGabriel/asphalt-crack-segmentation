import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def visualizar_mascara(ruta_imagen, ruta_mascara, ruta_salida=None,
                       color_mascara=(255, 0, 0), alfa=0.5,
                       mostrar=True, dpi=100):
    """
    Genera una visualización con tres paneles: imagen original, máscara y
    superposición.

    Args:
        ruta_imagen (str): Ruta a la imagen original
        ruta_mascara (str): Ruta a la máscara binaria
        ruta_salida (str, opcional): Ruta para guardar la visualización
        color_mascara (tuple): Color RGB para visualizar la máscara
        alfa (float): Transparencia para la superposición (0-1)
        mostrar (bool): Si se debe mostrar la imagen
        dpi (int): Resolución de la imagen de salida

    Returns:
        PIL.Image: Imagen combinada con los tres paneles
    """
    # Cargar imágenes
    imagen = Image.open(ruta_imagen).convert('RGB')
    mascara = Image.open(ruta_mascara).convert('L')

    # Asegurar que tienen el mismo tamaño
    if imagen.size != mascara.size:
        mascara = mascara.resize(imagen.size, Image.NEAREST)

    # Crear versión coloreada de la máscara para mejor visualización
    mascara_color = Image.new('RGB', mascara.size, (0, 0, 0))
    mascara_array = np.array(mascara)
    mascara_color_array = np.array(mascara_color)
    mascara_color_array[mascara_array > 127] = color_mascara
    mascara_color = Image.fromarray(mascara_color_array)

    # Crear superposición
    superposicion = imagen.copy()
    superposicion_array = np.array(superposicion)

    # Aplicar máscara con transparencia
    mask_indices = mascara_array > 127
    superposicion_array[mask_indices] = (
        (1 - alfa) * superposicion_array[mask_indices] +
        alfa * np.array(color_mascara)
    ).astype(np.uint8)

    superposicion = Image.fromarray(superposicion_array)

    # Crear imagen combinada horizontal (imagen | máscara | superposición)
    ancho, alto = imagen.size
    combinada = Image.new('RGB', (ancho * 3, alto))
    combinada.paste(imagen, (0, 0))
    combinada.paste(mascara_color, (ancho, 0))
    combinada.paste(superposicion, (ancho * 2, 0))

    # Añadir etiquetas
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    fig.sharey = True

    ax[0].imshow(np.array(imagen))
    ax[0].set_title('Imagen Original', size=20)
    ax[0].axis('on')

    ax[1].imshow(np.array(mascara), cmap='gray')
    ax[1].set_title('Máscara', size=20)
    ax[1].axis('on')

    ax[2].imshow(np.array(superposicion))
    ax[2].set_title('Superposición', size=20)
    ax[2].axis('on')

    plt.tight_layout()

    # Guardar o mostrar
    if ruta_salida:
        plt.savefig(ruta_salida, dpi=dpi, bbox_inches='tight')

    if mostrar:
        plt.show()
    else:
        plt.close()

    return combinada


# Ejemplo de uso
if __name__ == "__main__":
    # Rutas de ejemplo
    imagen_path = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
        doctorado/CrackDataset/luz_crack/images/4_1.jpg"
    mascara_path = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
        doctorado/CrackDataset/luz_crack/masks/4_1.jpg"
    salida_path = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
        doctorado/CrackDataset/luz_crack/verificacion/4_1_verificacion.png"

    # Crear visualización
    visualizar_mascara(imagen_path, mascara_path, salida_path)

    # Para procesar múltiples imágenes
    def procesar_lote(carpeta_imagenes, carpeta_mascaras, carpeta_salida):
        os.makedirs(carpeta_salida, exist_ok=True)

        # Obtener archivos de imágenes
        imagenes = [f for f in os.listdir(carpeta_imagenes)
                    if f.endswith(('.jpg', '.jpeg', '.png'))]

        for img_file in imagenes:
            nombre_base = os.path.splitext(img_file)[0]

            # Buscar la máscara correspondiente
            for ext in ['.png', '.jpg', '.jpeg']:
                mascara_file = nombre_base + ext
                mascara_path = os.path.join(carpeta_mascaras, mascara_file)
                if os.path.exists(mascara_path):
                    break
            else:
                print(f"No se encontró máscara para {img_file}")
                continue

            # Rutas completas
            img_path = os.path.join(carpeta_imagenes, img_file)
            salida_path = os.path.join(carpeta_salida, nombre_base +
                                       "_verificacion.png")

            # Generar visualización
            visualizar_mascara(img_path, mascara_path, salida_path,
                               mostrar=False)
            print(f"Procesada: {nombre_base}")

    # Ejemplo de procesamiento por lotes
    # procesar_lote("ruta/imagenes", "ruta/mascaras", "ruta/verificaciones")
