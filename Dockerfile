# Use NVIDIA CUDA 12.1 base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python, pip, and git
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -s /usr/bin/python3 /usr/bin/python

# Clone the repository
RUN git clone https://github.com/MattBeton/diloco-sim .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create data directory and copy the large file
RUN mkdir -p examples/data/owt
COPY examples/data/owt/openwebtext.bin examples/data/owt/openwebtext.bin

# Install the package in editable mode
RUN pip install -e .

# Set environment variables for NVIDIA runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Default command (can be overridden)
CMD ["python", "-c", "import torch; print(f'PyTorch {torch.__version__} available with CUDA: {torch.cuda.is_available()}')"]