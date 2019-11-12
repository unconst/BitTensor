FROM unconst/bittensor

# Copy across the local files.
COPY neurons/GoogleUSE bittensor/neurons/GoogleUSE

ENV TFHUB_CACHE_DIR /bittensor/cache

RUN pip install --upgrade pip && \
    pip install sentencepiece && \
    pip install tensorflow_hub==0.4.0 && \
    pip install tf-sentencepiece
