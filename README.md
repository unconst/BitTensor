## BitTensor Network Daemon

<img src="assets/mycellium.jpeg" width="1000" />

---

## Overview

BitTensor is neural network which trains across computers in a peer-to-peer fashion. In absence of centralized control, the network uses collaborative filtering to select computers with high informational significance. To these computers the network mints digital tokens which provide power over the network.

This repo contains an implementation of a peer in this network. It trains a self-supervised language representation using a dumpy corpus of text by taking as input the output of its peers in the network. In doing so, it mines the network native digital token.

For an in-depth description of this software, read the draft [paper](https://ipfs.eternum.io/ipfs/QmTUGaDaU2uSMTr2VEgbZ7JwBLokhnkXdvFPd14qSSYYRp/Bittensor.pdf)

---
## Run Locally
1. Install [python3](https://realpython.com/installing-python/)
1. Install [Docker](https://docs.docker.com/install/)

```
$ git clone https://github.com/unconst/BitTensor & cd BitTensor

# Start EOS chain.
$ ./start_eos.sh  

# Start node 1.
$ ./bittensor.sh

# Start node 2.
$ ./bittensor.sh
```
---

## Learn More

Join our [slack](https://bittensor.slack.com/) and say hello :)

---

## Pull Requests

This is alpha software, so in the interest of speed, just directly commit to the repo and use [Yapf](https://github.com/google/yapf) for code formatting.
```
$ pip install yapf
$ yapf --style google -r -vv -i .
```

---

## License

MIT
