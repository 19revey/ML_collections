FROM ubuntu:20.04
EXPOSE 8501

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    libgl1-mesa-glx libqt5core5a libqt5gui5 libqt5widgets5 \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
