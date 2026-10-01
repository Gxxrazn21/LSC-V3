"""Utilidades compartidas para el proyecto LSC.

Las dependencias de visualización se cargan bajo demanda para que las
herramientas de captura y validación no requieran Matplotlib al importarse.
"""

__all__ = [
    "graficar_curvas",
    "graficar_matriz_confusion",
    "evaluar_modelo",
    "reporte_clasificacion",
]


def __getattr__(name):
    if name in {"graficar_curvas", "graficar_matriz_confusion"}:
        from . import visualizacion
        return getattr(visualizacion, name)
    if name in {"evaluar_modelo", "reporte_clasificacion"}:
        from . import metricas
        return getattr(metricas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
