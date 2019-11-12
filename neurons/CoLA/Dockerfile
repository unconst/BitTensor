FROM unconst/bittensor:latest

# Copy across the local files.
COPY neurons/CoLA bittensor/neurons/CoLA

RUN pip install tensor2tensor
RUN pip3 install 'gast==0.2.2'
