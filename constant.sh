#!/usr/bin/env bash

DOCKER_IMAGE_NAME="bittensor"
DOCKER_IMAGE_TAG="latest"

function log() {
    python -c "from loguru import logger; logger.add(\"data/$IDENTITY/bittensor_logs.out\"); logger.debug(\"$1\")"
}

function success() {
    python -c "from loguru import logger; logger.add(\"data/$IDENTITY/bittensor_logs.out\"); logger.success(\"$1\")"
}

function failure() {
    python -c "from loguru import logger; logger.add(\"data/$IDENTITY/bittensor_logs.out\"); logger.error(\"$1\")"
}
