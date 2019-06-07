#!/usr/bin/env bash

DOCKER_IMAGE_NAME="bittensor"
DOCKER_IMAGE_TAG="latest"

function log() {
    python -c "from loguru import logger; logger.debug(\"$1\")"
}

function success() {
    python -c "from loguru import logger; logger.success(\"$1\")"
}

function failure() {
    python -c "from loguru import logger; logger.error(\"$1\")"
}
