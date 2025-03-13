import json
import matplotlib.pyplot as plt
import torch

# Cargar los datos de entrenamiento
with open('saved_models/model4/training_history.json', 'r') as f:
    history = json.load(f)

# Configurar el estilo para gráficas científicas de alta calidad
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12

# Obtener datos comunes y convertir a tensores para operaciones
epochs = torch.tensor(history['epochs'])
train_loss = torch.tensor(history['train_loss'])
val_loss = torch.tensor(history['val_loss'])
iou = torch.tensor(history['iou'])
f1 = torch.tensor(history['f1'])
precision = torch.tensor(history['precision'])
recall = torch.tensor(history['recall'])

# Estadísticas clave para impresión usando PyTorch
min_val_idx = torch.argmin(val_loss).item()
min_val = val_loss[min_val_idx].item()
min_epoch = epochs[min_val_idx].item()
max_iou_idx = torch.argmax(iou).item()
max_iou = iou[max_iou_idx].item()
max_epoch = epochs[max_iou_idx].item()

# 1. Crear y guardar gráfica de Pérdidas
plt.figure(figsize=(12, 8))
ax1 = plt.gca()
ax1.plot(epochs.numpy(), train_loss.numpy(), 'o-', color='#2C3E50',
         linewidth=2,
         label='Train Loss')
ax1.plot(epochs.numpy(), val_loss.numpy(), 'o-', color='#E74C3C', linewidth=2,
         label='Validation Loss')
ax1.set_title('Losses', fontsize=20, pad=15)
ax1.set_xlabel('Epoch', fontsize=16)  # Añadido eje X para gráfica individual
ax1.set_ylabel('Loss', fontsize=16)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(fontsize=16)

# Añadir anotación para el valor mínimo de validación
ax1.annotate(f'Mín: {min_val:.4f}',
             xy=(min_epoch, min_val), xytext=(-30, -30),
             textcoords='offset points', arrowprops=dict(arrowstyle='->'))

plt.tight_layout()
plt.savefig('saved_models/model4/training_losses.png', dpi=300,
            bbox_inches='tight')
plt.close()  # Cerrar la figura para liberar memoria

# 2. Crear y guardar gráfica de Métricas
plt.figure(figsize=(12, 8))
ax2 = plt.gca()
ax2.plot(epochs.numpy(), iou.numpy(), 'o-', color='#2980B9', linewidth=2,
         label='IoU')
ax2.plot(epochs.numpy(), f1.numpy(), 'o-', color='#27AE60', linewidth=2,
         label='F1')
ax2.plot(epochs.numpy(), precision.numpy(), 'o-', color='#8E44AD', linewidth=2,
         label='Precision')
ax2.plot(epochs.numpy(), recall.numpy(), 'o-', color='#F39C12', linewidth=2,
         label='Recall')
ax2.set_title('Metrics', fontsize=20, pad=15)
ax2.set_xlabel('Epoch', fontsize=16)
ax2.set_ylabel('Value', fontsize=16)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(fontsize=16)

# Añadir anotación para el valor máximo de IoU
ax2.annotate(f'IoU Máx: {max_iou:.4f}',
             xy=(max_epoch, max_iou), xytext=(0, 20),
             textcoords='offset points', arrowprops=dict(arrowstyle='->'))

plt.tight_layout()
plt.savefig('saved_models/model4/training_metrics.png', dpi=300,
            bbox_inches='tight')

# Opcional: mostrar la última gráfica (métricas)
plt.show()

# Imprimir estadísticas clave
print(f"Mejor IoU: {max_iou:.4f} (época {max_epoch})")
print(f"Mejor F1: {torch.max(f1).item():.4f} (época {epochs[torch.argmax(f1).
                                                            item()].item()})")
print(f"Menor pérdida de validación: {min_val:.4f} (época {min_epoch})")
