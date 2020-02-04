#!/bin/bash
import visualizer

import argparse
import grpc
import json
import pickle
import requests
import subprocess
import time

from config import Config
from concurrent import futures
from http import HTTPStatus
from loguru import logger
from TBLogger import TBLogger

class NodeStatsListener():

    def __init__(self, args):
        # Init configurations and arguments
        self._config = Config()
        self._args = args

        # Init server address.
        self._server_address = self._args.bind_address + ":" + self._args.port

        # Init node list.
        self._node_list = []

        # Map of node_id to channels.
        self._channels = {}

        # Map from node_id to logger.
        self._loggers = {}

        # Boolean visualizer is running.
        self._is_listening = True

    def __del__(self):
        logger.debug('Stop Listening at: {} ...', self._server_address)
        self._stop_listening()

    def _start_listening(self):
        self._is_listening = True
        self._run()

    def _stop_listening(self):
        self._is_listening = False

    def _run(self):
        if self.config.visualization_mode == "tensorboard":
            while self._is_listening:

                self._retrieve_all_nodes()
                self._refresh_all_channels()
                self._refresh_all_loggers()

                for node in self._nodes:
                    logger = self._loggers[node]
                    channel = self._channels[node]
                    response = self.query_node(node, channel)
                    self._log_response(response, logger)
        else:
            logger.info("Unexpected visualization_mode: {}", self.config.visualization_mode)

    def _retrieve_all_nodes(self):
        # Refresh list of nodes in the network
        self._node_list = []

        # Query the chain and retrieve all nodes that are "mining"
        payload = dict(
            code = self._config.eos_code,
            table = self._config.eos_table,
            scope = self._config.eos_scope,
            key_type = self._config.eos_key_type,
            json = "true"
        )

        payload_json = json.dumps(payload)
        try:
            request_url = self._config.eos_get_table_rows
            response = requests.post(url=request_url, data=payload_json)

            if response.status_code == HTTPStatus.OK:
                response_json = response.json()
                rows = response_json['rows']
                for row in rows:
                    node = dict(identity=row['identity'], url=row['address'], port=row['port'])
                    self._node_list.append(node)
            else:
                logger.error("Error: Could not retrieve the nodes connected to the chain.")

        except Exception as e:
            logger.exception("Failed to retrieve all node. Likely an issue connecting to the EOS chain", e)

        logger.info("Found {} nodes.".format(len(self._node_list)))

    def _refresh_all_channels(self):
        try:
            # Remove non-existent nodes.
            for node in self._channels.keys():
                if node not in self._node_list:
                    self._channels[node].close()
                    del self._channels[node]

            # Add new node channels.
            for node in self._node_list:
                if node not in self._channels.keys():
                    node_url = "{}:{}".format(node['url'], node['port'])
                    new_channel = grpc.insecure_channel(node_url)
                    self._channels[node] = new_channel

        except Exception as e:
            logger.exception("Failed refresh channels with exception: {}", e)

    def _refresh_all_loggers(self):
        try:
            # Remove non-existent nodes.
            for node in self._loggers.keys():
                if node not in self._node_list:
                    del self._loggers[node]

            # Add new node channels.
            for node in self._node_list:
                if node not in self._loggers.keys():
                    identity_dir = node['identity']
                    log_dir = self._args.logdir + "/" + identity_dir
                    self._loggers[node] = TBLogger(log_dir)
        except Exception as e:
            logger.exception("Failed refresh loggers with exception: {}", e)

    def _query_node(self, node, channel, logger):
        try:
            stub = visualizer.visualizer_pb2_grpc.VisualizerStub(channel)
            request_payload_bytes = pickle.dumps("tb_metrics", protocol=0)
            response = stub.Report(
                visualizer.visualizer_pb2.ReportRequest(
                    version = 1.0,
                    source_id = '2',
                    payload = request_payload_bytes
                )
            )
            # Let's process the incoming data
            response = pickle.loads(response.payload)
        except:
            logger.exception("Failed to query node: {}, with exception: {}", node, e)

        return response

    def _log_response(self, response, logger):
        try:
            step = response['step']
            logger.log_scalar("gs", response['gs'], step)
            logger.log_scalar("mem", response['mem'], step)
            logger.log_scalar("loss", response['loss'], step)

            # Metrics
            metrics = response['metrics']
            for idn in metrics.keys():
                logger.log_scalar(idn, metrics[idn], step)

            # Post scores to tb.
            scores = response['scores']
            if scores:
                for idn, score in scores:
                    logger.log_scalar(idn, score, step)

            logger.info('node {}: gs {} mem {} loss {} scores {}'.format(identity_dir, response['gs'], response['mem'], response['loss'], scores))
        except:
            logger.exception("Failed log response: {}, with exception: {}", response, e)


def main():
    listener = NodeStatsListener(args)
    listener._start_listening()

    try:
        while True:
            time.sleep(100)

    except KeyboardInterrupt:
        logger.info('Stopping visualizer with keyboard interrupt.')
        listener._stop_listening()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("bind_address", help="Address to which to bind the node listener gRPC server")
    parser.add_argument("port", help="Port that the stats listener will be listening on")
    parser.add_argument("logdir", help="Directory to which all tensorboard log files will be written.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
