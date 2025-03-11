# Documentación de Configuración para Segmentación de Grietas en Asfalto

Este documento describe la configuración utilizada para entrenar los modelos de segmentación de grietas en asfalto, con fines de reproducibilidad.

## Opciones de Arquitectura de Modelo

El sistema admite dos arquitecturas principales de modelo:

### 1. UNet (Estándar y con Dropout)

- Arquitectura U-Net básica con capas de dropout opcionales
- Canales de entrada: 3 (RGB)
- Canales de salida: 1 (segmentación binaria)
- Características: Codificador-decodificador estándar con conexiones residuales
- Configuración de dropout:
  - Dropout del codificador: 0.1 (aumenta con la profundidad, hasta 0.4)
  - Dropout del cuello de botella: 0.5
  - Dropout del decodificador: 0.1 (disminuye con el upsampling, hasta 0.025)

### 2. UNetResNet 

- Arquitectura U-Net con backbone ResNet
- Backbones disponibles: resnet18, resnet34, resnet50, resnet101, resnet152
- Transferencia de aprendizaje con pesos pre-entrenados de ImageNet
- Características: [256, 128, 64, 32]
- Dropout del decodificador: Tasa base de 0.1 con factor de 1.0
- Capacidad de descongelamiento gradual:
  - Comienza con el codificador congelado
  - Descongela capas progresivamente (desde las más profundas a las más superficiales)
  - Descongelamiento activado después de N épocas sin mejora del IoU
  - 5 etapas de descongelamiento (layer4 → layer3+4 → layer2+3+4 → etc.)

## Configuración de Entrenamiento

### Ajustes Generales

- Tamaño de lote (batch size): 32
- Tasa de aprendizaje: 0.001
- Decaimiento de pesos: 0.0001
- Número de épocas: 100
- Tamaño de imagen: 256×256
- Paciencia para parada temprana: 10 épocas
- Frecuencia de guardado de puntos de control: Cada 5 épocas
- Trabajadores: 14

### Configuración de Optimización

- Optimizador: Adam
- Planificador de tasa de aprendizaje: ReduceLROnPlateau
  - Modo: min (monitoreando pérdida de validación)
  - Factor: 0.5 (reduce a la mitad la tasa de aprendizaje)
  - Paciencia: 5 épocas
  - Tasa de aprendizaje mínima: 0.00001
- Pasos de acumulación de gradiente: 4
- Entrenamiento con precisión mixta: Habilitado

### Aumento de Datos

- Redimensionar a 256×256
- Volteo horizontal aleatorio
- Volteo vertical aleatorio
- Normalización con media/std de ImageNet: 
  - Media: [0.485, 0.456, 0.406]
  - Desviación estándar: [0.229, 0.224, 0.225]
- Aumentos específicos para entrenamiento:
  - Ruido aleatorio (opcional)
  - Sombras aleatorias (opcional)

## Métricas de Evaluación

Las siguientes métricas se calculan para la evaluación del modelo:
- IoU (Intersección sobre Unión)
- Precisión
- Exhaustividad (Recall)
- Puntuación F1

## Estrategia de Guardado del Modelo

- Mejor modelo por puntuación IoU
- Mejor modelo por pérdida de validación
- Modelo final en la última época
- Puntos de control regulares cada N épocas
- Cada punto de control guarda:
  - Pesos del modelo
  - Estado del optimizador
  - Estado del planificador
  - Número de época
  - Métricas en ese punto

## Requisitos de Hardware

El código admite:
- CUDA para aceleración GPU
- Fallback a CPU cuando la GPU no está disponible
- Precisión mixta para mejor rendimiento en GPUs compatibles

## Notas Adicionales

- División del conjunto de datos: 70% entrenamiento, 15% validación, 15% prueba
- Semilla aleatoria 42 utilizada para reproducibilidad
- Registro con TensorBoard habilitado para monitoreo
