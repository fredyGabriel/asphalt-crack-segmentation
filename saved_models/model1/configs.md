# Resumen de Configuraciones y Parámetros

## Archivo default.yaml (Configuración General):

- Paths:
    - data_path: ruta base de datos.
    - dataset_path: ruta específica al conjunto de datos (por ejemplo, "CrackDataset/luz_crack/").

- Training:
    - batch_size: 32
    - learning_rate: 0.001
    - num_epochs: 100
    - weight_decay: 0.0001
    - num_workers: 12
    - save_frequency: guarda checkpoints cada 5 épocas
    - early_stopping_patience: 10

- Model:
    - input_channels: 3
    - output_channels: 1
    - num_filters: 64
    - save_path: "saved_models/" (carpeta donde se guardan los modelos y logs)

- Evaluation:
    - Métricas a calcular: iou, recall, precision, f1
    - Guardar resultados en "results/evaluation_results.json"

- Archivo unet.py (Definición del Modelo UNet):
    - Arquitectura basada en bloques de “DoubleConv” y “DownSample/UpSample”.
    - Parámetros de entrada: 3 canales (RGB).
    - Salida: 1 canal (para segmentación binaria).
    - La cantidad inicial de filtros es 64 y se duplica o se reduce en cada bloque según la estructura típica de UNet.

- Archivo train.py (Entrenamiento y Logging):
    - Se usan técnicas como entrenamiento con precisión mixta (GradScaler y autocast).
    - Se emplea un optimizador Adam con el weight_decay definido en default.yaml.
    - Se utiliza un scheduler (ReduceLROnPlateau) para ajustar la tasa de aprendizaje basándose en la pérdida de validación.
    - Se guarda el historial de pérdidas y métricas en un diccionario y se loguea a - TensorBoard (los logs se guardan en la carpeta indicada por config['model']['save_path']).
    - Se implementa Early Stopping: si la pérdida de validación no mejora en "early_stopping_patience" épocas, se detiene el entrenamiento.
    - Además, se guardan checkpoints y se realiza el guardado especial del “mejor modelo” tanto por pérdida como por la métrica IoU.