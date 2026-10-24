# Dockerfile para Q-Rutas Piura
# Entorno reproducible con Python 3.11, dependencias científicas, Qiskit y Gradio.

FROM python:3.11-slim

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Instalar dependencias del sistema operativo
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar requerimientos e instalar dependencias de Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gradio pydantic

# Copiar el código fuente
COPY . /app/

# Exponer el puerto para la demo web de Gradio
EXPOSE 7860

# Comando por defecto: lanzar la demo interactiva
CMD ["python", "app_demo.py"]
