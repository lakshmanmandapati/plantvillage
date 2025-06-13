# Use a slim Python base image for smaller image size.
# Python 3.9 is chosen for broad compatibility with TensorFlow.
# Changed from 'python:3.9.slim' to 'python:3.9-slim-buster' for better availability.
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for TensorFlow (e.g., for numpy, scipy)
# and for other potential Python packages.
# The `rm -rf /var/lib/apt/lists/*` cleans up the apt cache to keep the image size down.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the working directory
COPY requirements.txt .

# Install the Python dependencies
# Use --no-cache-dir to prevent pip from caching packages, further reducing image size.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the 'models' directory and its contents into the container
# This ensures your models and class name files are available
COPY models/ ./models/

# Copy the Streamlit application file into the container
COPY app.py .

# Expose the port that Streamlit runs on (default is 8501)
EXPOSE 8501

# Define the command to run the Streamlit application
# The '--server.port' and '--server.address' arguments ensure Streamlit
# listens on all available network interfaces within the container, making it
# accessible from outside the container.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
