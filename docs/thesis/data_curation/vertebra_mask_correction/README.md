# Documentación de corrección y expansión de máscaras vertebrales

Esta carpeta contiene la documentación técnica del proceso de curación de máscaras utilizado en el proyecto de segmentación vertebral T1–L5.

El documento describe dos fases principales:

1. Corrección de máscaras multicategoría existentes mediante conversión ID → RGB, edición manual en GIMP, validación estricta de colores y reconversión RGB → ID.
2. Generación de pre-máscaras binarias para apoyar la expansión manual del dataset usando nuevas imágenes candidatas.

## Archivos

- `documentacion_correccion_mascaras_vertebrales.pdf`: documentación completa del flujo de corrección, ejemplos visuales, criterios de decisión, limitaciones y guía práctica.

## Alcance

Este proceso corresponde a una curación técnica y visual del dataset. Las correcciones mejoran la consistencia de las máscaras para entrenamiento/evaluación, pero no deben interpretarse como una validación clínica definitiva.

Las pre-máscaras binarias generadas para nuevas imágenes son insumos editables de anotación asistida. No se consideran ground truth final hasta ser revisadas y corregidas manualmente.

## Relación con el artículo

Este material se referencia en el artículo como documentación complementaria del proceso de curación de datos, expansión del dataset y trazabilidad de las decisiones tomadas durante la preparación de las máscaras.