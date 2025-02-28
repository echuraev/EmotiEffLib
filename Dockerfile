# This docker contains prepared environment for building binder docker container
FROM       ubuntu:jammy
CMD        bash

# Set noninteractive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Required system packages
RUN apt-get update && apt-get -y install \
    cmake \
    g++ \
    make \
    wget \
    libopencv-dev

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/miniconda && \
    rm /miniconda.sh

# Set up Conda paths
ENV PATH="/opt/miniconda/bin:$PATH"

# Copy the environment.yml file
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml

# Activate the environment in the container
SHELL ["conda", "run", "-n", "emotiefflib", "/bin/bash", "-c"]

# Copy requirements files
RUN mkdir -p /requirements/tests
COPY requirements*.txt /requirements/
COPY tests/requirements.txt /requirements/tests/

# Install dependencies
RUN python -m pip install --upgrade pip && \
    pip install -r /requirements/requirements.txt && \
    pip install -r /requirements/requirements-engagement.txt && \
    pip install -r /requirements/requirements-torch.txt && \
    pip install -r /requirements/tests/requirements.txt
