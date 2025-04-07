FROM balenalib/raspberrypi4-64-python:3.9

WORKDIR /app

# Install dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libatlas-base-dev \
    python3-opencv

# Install Python packages
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Copy model and script
COPY . .

# Run the script
CMD ["python3", "drowning_detection.py"]
