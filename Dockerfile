# Base image with CUDA support
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uninstall CPU-only PyTorch
RUN pip uninstall -y torch torchvision torchaudio

# Install GPU-compatible PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Load and cache models, FAISS index, and embeddings during the build process
COPY resources.py ./
RUN python3 resources.py --prepare

# Copy application files
COPY . .

# Expose port
EXPOSE 5000

# Run the Flask app
CMD ["python3", "app.py"]
