# =============================================================
# Dockerfile - LSC v3.0 (Servidor de Reconocimiento y Pruebas Móviles)
# =============================================================
FROM python:3.11-slim

# Variables de entorno para Python optimizado en contenedor
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Instalar dependencias del sistema necesarias para MediaPipe y OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar requerimientos de Python en capa cacheada
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el proyecto
COPY . .

# Exponer puerto para pruebas web móviles
EXPOSE 8000

# Iniciar servidor Uvicorn accesible desde cualquier dispositivo en la red
CMD ["python", "-m", "uvicorn", "motor_lsc.servidor_app:app", "--host", "0.0.0.0", "--port", "8000"]
