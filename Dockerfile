#FROM custom-cuda-python:11.2.2
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
#FROM python:3.8-slim

WORKDIR /app/
# Install necessary tools
RUN apt-get update && \
    apt-get install -y wget unzip curl git python3-pip python3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN pip3 install --upgrade pip setuptools wheel

# Install specific versions of Cython and other dependencies
#RUN pip3 install cython==0.29.28

# Copy requirements files
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Convert the Jupyter notebook to a Python script and run it
RUN pip3 install nbconvert

COPY . /app/

RUN jupyter nbconvert --to script Fastformer.ipynb
RUN ls -l /app && cat /app/Fastformer.py && echo "current dir: $(pwd)"
RUN chmod +x Fastformer.py
RUN chmod 600 Fastformer.py
RUN ls -l /app && cat /app/Fastformer.py && echo "current dir: $(pwd)"
# Set the command to run the converted Python script
CMD ["jupyter nbconvert --to script Fastformer.ipynb || ls -l /app || python3", "/app/Fastformer.py"]