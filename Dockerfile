FROM eosio-bittensor:latest

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
  pip install libeospy && \
  pip install loguru && \
  pip install matplotlib && \
  pip install numpy && \
  pip install sklearn && \
  pip install tensorflow && \
  pip install zipfile36

# Copy BitTensor source to this image.
RUN mkdir bittensor
COPY . bittensor/
WORKDIR /bittensor

RUN pip install --upgrade pip && pip install -r requirements.txt
