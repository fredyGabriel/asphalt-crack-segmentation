import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def convert_jpg_to_png(input_folder, output_folder=None,
                       preserve_structure=True):
    """
    Convierte todas las imágenes JPG de una carpeta a formato PNG.

    Args:
        input_folder (str): Ruta a la carpeta con las imágenes JPG
        output_folder (str, optional): Ruta donde se guardarán imágenes PNG.
                                     Si es None, se crea carpeta 'png_images'
                                     en el mismo nivel que input_folder.
        preserve_structure (bool): Si es True, mantiene la estructura de
            subcarpetas
    """
    # Normalizar rutas
    input_folder = Path(input_folder)

    # Crear carpeta de salida si no se especifica
    if output_folder is None:
        output_folder = input_folder.parent / f"{input_folder.name}_png"
    else:
        output_folder = Path(output_folder)

    # Verificar que la carpeta de entrada exista
    if not input_folder.exists():
        raise FileNotFoundError(f"La carpeta {input_folder} no existe")

    # Crear carpeta de salida
    output_folder.mkdir(exist_ok=True, parents=True)

    # Contadores para estadísticas
    total_converted = 0
    errors = 0

    # Extensiones a convertir (incluye variaciones de capitalización)
    jpg_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG')

    # Encontrar todas las imágenes JPG recursivamente
    all_images = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endsWith(jpg_extensions):
                all_images.append(os.path.join(root, file))

    # Procesar imágenes con barra de progreso
    print(f"Convirtiendo {len(all_images)} imágenes...")
    for img_path in tqdm(all_images):
        # Determinar la ruta de salida
        rel_path = os.path.relpath(img_path, input_folder)

        if preserve_structure:
            # Mantener estructura de carpetas
            out_path = output_folder / Path(rel_path).with_suffix('.png')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        else:
            # Guardar todas las imágenes en la raíz de output_folder
            out_path = output_folder / f"{Path(img_path).stem}.png"

        try:
            # Abrir imagen y convertirla a tensor para procesamiento
            with Image.open(img_path) as img:
                img.save(out_path, "PNG")
            total_converted += 1
        except Exception as e:
            print(f"Error al convertir {img_path}: {e}")
            errors += 1

    # Mostrar resultados
    print("\nConversión completa:")
    print(f"- Total convertidas: {total_converted}")
    print(f"- Errores: {errors}")
    print(f"- Imágenes guardadas en: {output_folder}")

    return output_folder


if __name__ == "__main__":
    inputs = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
        CrackDataset/luz_crack/masks/"
    outputs = "C:/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/\
        CrackDataset/luz_crack/masks_png/"

    # Ejecutar conversión
    convert_jpg_to_png(inputs, outputs)
