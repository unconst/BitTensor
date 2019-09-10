FROM unconst/bittensor

# Copy BitTensor source to this image.
RUN mkdir bittensor
COPY . bittensor/
WORKDIR /bittensor
