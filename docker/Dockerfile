FROM ubuntu:18.04

RUN apt-get update && apt-get install -y wget sudo curl
RUN wget https://github.com/EOSIO/eosio.cdt/releases/download/v1.6.1/eosio.cdt_1.6.1-1_amd64.deb
RUN apt-get update && sudo apt install -y ./eosio.cdt_1.6.1-1_amd64.deb
RUN wget https://github.com/eosio/eos/releases/download/v1.8.1/eosio_1.8.1-1-ubuntu-18.04_amd64.deb
RUN apt-get update && sudo apt install -y ./eosio_1.8.1-1-ubuntu-18.04_amd64.deb

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
        python \
        python-dev \
        python-setuptools \
        python-pip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
  pip install argparse && \
  pip install grpcio && \
  pip install grpcio-tools && \
  pip install libeospy && \
  pip install loguru && \
  pip install matplotlib && \
  pip install "numpy<1.17" && \
  pip install networkx && \
  pip install pebble && \
  pip install pickle-mixin && \
  pip install tensorflow && \
  pip install timeloop && \
  pip install zipfile36
