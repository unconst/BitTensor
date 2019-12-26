#!/usr/bin/env bash

DOCKER_IMAGE_NAME="bittensor"
DOCKER_IMAGE_TAG="latest"

function trace() {
    python3 -c "from loguru import logger; logger.add(\"data/$IDENTITY/bittensor_logs.out\"); logger.debug(\"$1\")" > /dev/null 2>&1
}

function log() {
    python3 -c "from loguru import logger; logger.add(\"data/$IDENTITY/bittensor_logs.out\"); logger.debug(\"$1\")"
}

function success() {
    python3 -c "from loguru import logger; logger.add(\"data/$IDENTITY/bittensor_logs.out\"); logger.success(\"$1\")"
}

function failure() {
    python3 -c "from loguru import logger; logger.add(\"data/$IDENTITY/bittensor_logs.out\"); logger.error(\"$1\")"
}

function whichmachine() {
    unameOut="$(uname -s)"
    case "${unameOut}" in
        Linux*)     machine=Linux;;
        Darwin*)    machine=Mac;;
        CYGWIN*)    machine=Cygwin;;
        MINGW*)     machine=MinGw;;
        *)          machine="UNKNOWN:${unameOut}"
    esac
    echo ${machine}
}
