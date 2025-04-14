FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    ffmpeg \
    unzip \
    python3.10 \
    python3-pip

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN pip install gdown

RUN mkdir -p /results

RUN gdown https://drive.google.com/uc?id=1Z_20gn51X2OPPcNdE2HEkPaUWJ0I4-dh -O /tmp/file.zip && \
    unzip /tmp/file.zip -d /results && \
    rm /tmp/file.zip \

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r planner-requirements.txt

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run Jupyter Notebook when the container launches
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]