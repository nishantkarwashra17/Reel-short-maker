FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    imagemagick \
    fonts-liberation \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Fix ImageMagick policy - allow all operations
RUN if [ -f /etc/ImageMagick-6/policy.xml ]; then \
    sed -i 's/rights="none"/rights="read|write"/g' /etc/ImageMagick-6/policy.xml; \
    fi

# Hugging Face requires a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# Install Python deps first (cache layer)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy app code
COPY --chown=user . .

# Create downloads directory
RUN mkdir -p downloads

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0", "--server.fileWatcherType", "none"]
