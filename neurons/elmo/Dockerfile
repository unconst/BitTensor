FROM unconst/bittensor

# Copy across the local files.
COPY . bittensor/

ENV TFHUB_CACHE_DIR /bittensor/cache

RUN pip install --upgrade pip && \
    pip install tensorflow_hub==0.4.0
