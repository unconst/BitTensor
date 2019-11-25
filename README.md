## BitTensor Network Daemon

<img src="assets/mycellium.jpeg" width="1000" />

---

## Overview

BitTensor is a machine intelligence system which can harness any internet connected computer. In absence of centralized control, that network uses a digital token to provide incentive for intelligence contribution where the reciever has more sway over the network's mathematical objective. This repo contains the open source software which runs that resulting network and mines that token. 

For an in-depth description of this software, see https://www.bittensor.com/learn

---
## Install and Run (Testnet)

1. [Install Docker](https://docs.docker.com/install/)
1. Then
```
$ git clone https://github.com/unconst/BitTensor & cd BitTensor
$ ./bittensor.sh --upnpc --eosurl http://159.65.102.106:8888
```
---

## Learn More

Read the [paper](https://www.bittensor.com/learn) or join our [slack](https://bittensor.slack.com/)

---


## Pull Requests

This is alpha software, so in the interest of speed, just directly commit to the repo. To make that feasible, try to keep your work as modular as possible. I like to iterate fast by creating another sub project where tests can grow. For instance, in this repo, the sync_kgraph, and async_kgraph are separate independent implementations. Yes this creates code copying and rewrite, but allows fast development.

Also, use [Yapf](https://github.com/google/yapf) for code formatting. You can run the following to format before a commit.
```
$ pip install yapf
$ yapf --style google -r -vv -i .
```

---

## License

MIT
