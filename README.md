# Glitchy Silhouette

Detección de siluetas en tiempo real con efectos glitch y modo cabina de fotos.

## Instalación

```bash
git clone https://github.com/tomas-b/glitchy-pose-silhouette.git
cd glitchy-pose-silhouette
uv sync
uv run python glitchy_silhouette.py
```

## Uso

1. Párate frente a la cámara
2. Espera la cuenta regresiva de 5 segundos
3. Muévete para ver efectos glitch en tu silueta
4. Levanta la mano sobre tu cabeza para tomar una foto

## Controles

- **ESPACIO** - Cambiar efecto
- **W/S** - Ajustar sensibilidad  
- **A/D** - Ajustar intensidad
- **R** - Reiniciar
- **Q** - Salir

## Cabina de Fotos

Cuando levantas la mano:
- Aparece un recuadro rojo con cuenta regresiva de 5 segundos
- Flash blanco = foto tomada
- Guarda dos imágenes: original y con efectos
- 5 segundos de espera antes de la siguiente foto

## Efectos

- Píxeles Aleatorios - Ruido colorido
- Bloques Glitch - Distorsiones rectangulares  
- Líneas de Escaneo - Interferencia horizontal
- Datamosh - Artefactos de compresión

## Detalles Técnicos

- Usa OpenCV para detección de movimiento (sustracción de fondo MOG2)
- MediaPipe para tracking de pose/esqueleto
- Los efectos solo se aplican dentro del área del cuerpo detectado para mejor rendimiento
- Captura de cámara con threads para operación fluida

## Requisitos

- Python 3.8+
- Webcam
- Gestor de paquetes uv