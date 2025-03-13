import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def resaltar_fisuras_kernel_promedio(imagen_gris, kernel_size=3,
                                     rango_fisura=(80, 140),
                                     umbral_nofisura=150,
                                     factor_aclarar=1.2,
                                     factor_oscurecer=0.5,
                                     use_vectorized=True):
    """
    Resalta fisuras en imágenes de pavimento usando un enfoque de promedio
        local
    y transformaciones condicionales de intensidad.

    Parámetros:
    -----------
    imagen_gris : ndarray
        Imagen en escala de grises como array numpy
    kernel_size : int
        Tamaño del kernel cuadrado para calcular promedios locales
    rango_fisura : tuple
        Rango (min, max) de intensidades promedio que indican posible fisura
    umbral_nofisura : int
        Umbral por encima del cual se considera "no fisura"
    factor_aclarar : float
        Factor para aclarar zonas de "no fisura" (>1.0)
    factor_oscurecer : float
        Factor para oscurecer zonas de "fisura" (<1.0)
    use_vectorized : bool
        Si es True, usa operaciones vectorizadas para mayor eficiencia

    Returns:
    --------
    ndarray
        Imagen con fisuras resaltadas
    """
    # Verificar que la imagen está en escala de grises
    if len(imagen_gris.shape) > 2:
        raise ValueError("La imagen debe estar en escala de grises")

    # Crear kernel de promedio normalizado
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (
        kernel_size * kernel_size)

    # Calcular imagen de promedios locales
    promedios_locales = cv2.filter2D(imagen_gris, -1, kernel)

    # Crear una copia de la imagen original como float32 para manipulaciones
    imagen_resultado = imagen_gris.astype(np.float32)

    if use_vectorized:
        # Enfoque vectorizado (más eficiente)
        # Crear máscaras para zonas de fisura y no-fisura
        mascara_nofisura = ((promedios_locales < rango_fisura[0]) |
                            (promedios_locales > rango_fisura[1]) |
                            (promedios_locales > umbral_nofisura))

        mascara_fisura = ~mascara_nofisura

        # Aplicar transformaciones
        imagen_resultado[mascara_nofisura] = \
            imagen_resultado[mascara_nofisura] * factor_aclarar
        imagen_resultado[mascara_fisura] = imagen_resultado[mascara_fisura] * \
            factor_oscurecer
    else:
        # Enfoque no vectorizado (iterativo)
        altura, anchura = imagen_gris.shape

        # Iterar sobre cada píxel de la imagen
        for y in range(altura):
            for x in range(anchura):
                promedio_local = promedios_locales[y, x]

                # Condición "No Fisura"
                if (promedio_local < rango_fisura[0] or
                    promedio_local > rango_fisura[1] or
                        promedio_local > umbral_nofisura):
                    # Aclarar
                    imagen_resultado[y, x] = imagen_gris[y, x] * factor_aclarar
                else:
                    # Condición "Fisura" - Oscurecer
                    imagen_resultado[y, x] = imagen_gris[y, x] *\
                        factor_oscurecer

    # Asegurar rango válido [0, 255]
    imagen_resultado = np.clip(imagen_resultado, 0, 255).astype(np.uint8)

    return imagen_resultado


def visualizar_comparacion(imagen_original, imagen_resaltada,
                           titulo="Comparación de Resaltado de Fisuras"):
    """
    Visualiza la imagen original y la imagen con fisuras resaltadas lado a lado

    Parámetros:
    -----------
    imagen_original : ndarray
        Imagen original en escala de grises
    imagen_resaltada : ndarray
        Imagen con fisuras resaltadas
    titulo : str
        Título para la figura
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(imagen_original, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(imagen_resaltada, cmap='gray')
    plt.title('Fisuras Resaltadas')
    plt.axis('off')

    plt.suptitle(titulo)
    plt.tight_layout()
    plt.show()

    return plt.gcf()  # Devolver la figura para posible guardado


def explorar_parametros(imagen_gris, output_dir=None, guardar_resultados=False
                        ):
    """
    Explora diferentes combinaciones de parámetros para el resaltado de fisuras

    Parámetros:
    -----------
    imagen_gris : ndarray
        Imagen en escala de grises
    output_dir : str or None
        Directorio para guardar resultados si guardar_resultados=True
    guardar_resultados : bool
        Si es True, guarda las imágenes procesadas
    """
    # Crear directorio de salida si es necesario
    if guardar_resultados and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Definir algunas combinaciones específicas para exploración
    combinaciones = [
        # kernel, rango_fisura, umbral_nofisura, factor_aclarar,
        # factor_oscurecer
        (3, (80, 140), 150, 1.2, 0.8),  # Base case
        (5, (80, 140), 150, 1.2, 0.8),  # Mayor kernel
        (3, (70, 130), 150, 1.2, 0.8),  # Rango más bajo
        (3, (90, 150), 150, 1.2, 0.8),  # Rango más alto
        (3, (80, 140), 140, 1.2, 0.8),  # Umbral más bajo
        (3, (80, 140), 160, 1.2, 0.8),  # Umbral más alto
        (3, (80, 140), 150, 1.3, 0.8),  # Mayor contraste (aclarar)
        (3, (80, 140), 150, 1.2, 0.7),  # Mayor contraste (oscurecer)
    ]

    resultados = []

    # Procesar cada combinación
    for idx, (k, r, u, fa, fo) in enumerate(combinaciones):
        # Aplicar el resaltado
        imagen_resaltada = resaltar_fisuras_kernel_promedio(
            imagen_gris,
            kernel_size=k,
            rango_fisura=r,
            umbral_nofisura=u,
            factor_aclarar=fa,
            factor_oscurecer=fo
        )

        # Guardar resultado
        resultados.append({
            'imagen': imagen_resaltada,
            'kernel_size': k,
            'rango_fisura': r,
            'umbral_nofisura': u,
            'factor_aclarar': fa,
            'factor_oscurecer': fo,
            'descripcion': f"K={k}, R={r}, U={u}, FA={fa}, FO={fo}"
        })

        if guardar_resultados and output_dir:
            nombre_archivo = f"resaltado_k{k}_r{r[0]}-{r[1]}_u{u}_fa{fa}\
_fo{fo}.png"
            ruta_completa = os.path.join(output_dir, nombre_archivo)
            cv2.imwrite(ruta_completa, imagen_resaltada)

    # Visualizar resultados en una cuadrícula
    n = len(resultados)
    filas = int(np.ceil(n / 3))
    cols = min(n, 3)

    plt.figure(figsize=(15, filas * 4))

    # Mostrar imagen original
    plt.subplot(filas, cols, 1)
    plt.imshow(imagen_gris, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # Mostrar resultados
    for i, res in enumerate(resultados):
        if i == 0:
            continue  # Skip the first subplot since it's the original
        plt.subplot(filas, cols, i + 1)
        plt.imshow(res['imagen'], cmap='gray')
        plt.title(f"Combinación {i}\n{res['descripcion']}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    if guardar_resultados and output_dir:
        plt.savefig(os.path.join(output_dir, "comparacion_parametros.png"),
                    dpi=300, bbox_inches='tight')

    return resultados


def process_directory(input_dir, output_dir, file_pattern="*.jpg",
                      kernel_size=3, rango_fisura=(80, 140),
                      umbral_nofisura=150, factor_aclarar=1.2,
                      factor_oscurecer=0.8):
    """
    Procesa todas las imágenes en un directorio utilizando el método de
    resaltado de fisuras.

    Parámetros:
    -----------
    input_dir : str
        Directorio con las imágenes originales
    output_dir : str
        Directorio para guardar las imágenes procesadas
    file_pattern : str
        Patrón para encontrar archivos (por defecto "*.jpg")
    kernel_size, rango_fisura, umbral_nofisura, factor_aclarar,
        factor_oscurecer:
        Parámetros para la función de resaltado de fisuras
    """
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Encontrar archivos que coincidan con el patrón
    input_path = Path(input_dir)
    archivos = list(input_path.glob(file_pattern))

    if not archivos:
        print(f"No se encontraron archivos que coincidan con \
{file_pattern} en {input_dir}")
        return

    # Procesar cada archivo
    for archivo in tqdm(archivos, desc="Procesando imágenes"):
        try:
            # Leer imagen
            imagen = cv2.imread(str(archivo), cv2.IMREAD_GRAYSCALE)
            if imagen is None:
                print(f"No se pudo leer {archivo}")
                continue

            # Aplicar resaltado de fisuras
            imagen_resaltada = resaltar_fisuras_kernel_promedio(
                imagen,
                kernel_size=kernel_size,
                rango_fisura=rango_fisura,
                umbral_nofisura=umbral_nofisura,
                factor_aclarar=factor_aclarar,
                factor_oscurecer=factor_oscurecer
            )

            # Guardar resultado
            ruta_salida = os.path.join(output_dir,
                                       f"{archivo.stem}_resaltado.png")
            cv2.imwrite(ruta_salida, imagen_resaltada)

        except Exception as e:
            print(f"Error al procesar {archivo}: {str(e)}")

    print(f"Se procesaron {len(archivos)} imágenes. Resultados guardados en \
{output_dir}")


def generar_mascara_otsu(imagen_resaltada):
    """
    Genera una máscara binaria a partir de una imagen resaltada usando
    umbralización de Otsu.

    Parámetros:
    -----------
    imagen_resaltada : ndarray
        Imagen con fisuras resaltadas

    Returns:
    --------
    ndarray
        Máscara binaria donde 255 representa posibles fisuras
    """
    # Aplicar umbralización de Otsu
    _, mascara = cv2.threshold(imagen_resaltada, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Opcional: aplicar operaciones morfológicas para mejorar la máscara
    kernel = np.ones((3, 3), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mascara


def resaltar_y_segmentar(imagen_gris, kernel_size=3,
                         rango_fisura=(80, 140),
                         umbral_nofisura=150,
                         factor_aclarar=1.2,
                         factor_oscurecer=0.8):
    """
    Resalta fisuras y genera una segmentación automática.

    Parámetros:
    -----------
    imagen_gris : ndarray
        Imagen en escala de grises
    kernel_size, rango_fisura, umbral_nofisura, factor_aclarar,
        factor_oscurecer:
        Parámetros para la función de resaltado de fisuras

    Returns:
    --------
    tuple
        (imagen_resaltada, mascara) - Imagen resaltada y máscara binaria
    """
    # Aplicar el resaltado de fisuras
    imagen_resaltada = resaltar_fisuras_kernel_promedio(
        imagen_gris,
        kernel_size=kernel_size,
        rango_fisura=rango_fisura,
        umbral_nofisura=umbral_nofisura,
        factor_aclarar=factor_aclarar,
        factor_oscurecer=factor_oscurecer
    )

    # Generar máscara usando Otsu
    mascara = generar_mascara_otsu(imagen_resaltada)

    return imagen_resaltada, mascara


def visualizar_resultados_completos(imagen_original, imagen_resaltada, mascara
                                    ):
    """
    Visualiza la imagen original, la imagen resaltada y la máscara generada.

    Parámetros:
    -----------
    imagen_original : ndarray
        Imagen original en escala de grises
    imagen_resaltada : ndarray
        Imagen con fisuras resaltadas
    mascara : ndarray
        Máscara binaria resultante de la segmentación
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(imagen_original, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(imagen_resaltada, cmap='gray')
    plt.title('Fisuras Resaltadas')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(mascara, cmap='gray')
    plt.title('Máscara Binaria')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Ejemplo de uso con una imagen
    imagen_path = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/images/15_1.jpg"
    output_dir = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
CrackDataset/luz_crack/local_average_enhancement/"

    # Cargar imagen en escala de grises
    imagen_gris = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

    if imagen_gris is None:
        print(f"No se pudo cargar la imagen: {imagen_path}")
    else:
        # Aplicar el método de resaltado de fisuras con parámetros por defecto
        print("Aplicando resaltado de fisuras con parámetros por defecto...")
        imagen_resaltada = resaltar_fisuras_kernel_promedio(imagen_gris)

        # Visualizar resultados
        fig = visualizar_comparacion(imagen_gris, imagen_resaltada)

        # Guardar resultado
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "15_1_resaltado.png"),
                    imagen_resaltada)

        # Explorar diferentes parámetros (opcional)
        print("Explorando diferentes combinaciones de parámetros...")
        explorar_parametros(imagen_gris, output_dir, guardar_resultados=True)

        # Segmentación automática
        print("Aplicando resaltado y segmentación automática...")
        imagen_resaltada, mascara = resaltar_y_segmentar(imagen_gris)
        visualizar_resultados_completos(imagen_gris, imagen_resaltada, mascara)

        # Guardar máscara
        cv2.imwrite(os.path.join(output_dir, "15_1_mascara.png"), mascara)

        print("Procesamiento completado. Resultados guardados en:", output_dir)

        # Ejemplo de procesamiento de directorio (comentado)
        """
        process_directory(
            input_dir="C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
doctorado/CrackDataset/luz_crack/images/",
            output_dir="C:/Users/fgrv/OneDrive/Documentos/PythonProjects/\
doctorado/CrackDataset/luz_crack/local_average_enhancement/",
            kernel_size=3,
            rango_fisura=(80, 140),
            umbral_nofisura=150,
            factor_aclarar=1.2,
            factor_oscurecer=0.8
        )
        """
