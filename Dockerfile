FROM unconst/bittensor

# Copy BitTensor source to this image.
COPY . .

RUN pip install -e .
