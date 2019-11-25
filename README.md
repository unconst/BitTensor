## BitTensor Network Daemon

<img src="assets/mycellium.jpeg" width="1000" />

---

## Overview

A machine intelligence trained with access to the internet could harness resources of data, machine knowledge and computer power untapped by its predecessors.

This repo contains a protocol implementation for a trustless, decentralized and incentivised peer-to-peer model that can connect an unlimited number of computers and engineers across the web.

In absence of centralized control, the network uses a digital token to provide incentive for intelligence contribution. The reciever has more sway over the network's mathematical objective.

Running this software connects your computer to this network. Unless otherwise changed, your node trains on a corpus of unlabaled langauge to train a multi-task language embedding.

---

## Install and Run

To begin, you will need to:

1. [Install Docker](https://docs.docker.com/install/)

Then:
```
$ git clone https://github.com/unconst/BitTensor & cd BitTensor
$ ./bittensor.sh --upnpc --eosurl http://159.65.102.106:8888
```
---

## Learn More

Read the [paper](https://www.bittensor.com/) or join our [slack](https:// channnel.

## Pull Requests

In the interest of speed, just directly commit to the repo. To make that feasible, try to keep your work as modular as possible. I like to iterate fast by creating another sub project where tests can grow. For instance, in this repo, the sync_kgraph, and async_kgraph are separate independent implementations. Yes this creates code copying and rewrite, but allows fast development.

Also, use [Yapf](https://github.com/google/yapf) for code formatting. You can run the following to format before a commit.
```
$ pip install yapf
$ yapf --style google -r -vv -i .
```

MIT
