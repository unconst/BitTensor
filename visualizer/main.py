#!/bin/bash
import visualizer

import argparse
import grpc
import json
import pickle
import requests
import subprocess
import time
import yaml

from config import Config
from concurrent import futures
from http import HTTPStatus
from loguru import logger
from TBLogger import TBLogger

class NodeStatsListener():

    def __init__(self, config):
        # Init configurations and arguments
        self._config = config

        # Map from node_id to node attr dict.
        self._nodes = {}

        # Map of node_id to channels.
        self._channels = {}

        # Map from node_id to logger.
        self._tbloggers = {}

        # Global step counter.
        self._global_step = 0

    def refresh(self):
        logger.info('Refresh')

        # Refresh state.
        try:
            self._retrieve_all_nodes()
        except Exception as e:
            logger.exception("Failed to retrieve all node. Likely an issue connecting to the EOS chain", e)

        try:
            self._refresh_all_channels()
        except Exception as e:
            logger.exception("Failed refresh channels with exception: {}", e)

        try:
            self._refresh_all_tbloggers()
        except Exception as e:
            logger.exception("Failed refresh loggers with exception: {}", e)

        # Make queries.
        if self._config.visualization_mode != "tensorboard":
            logger.info("Unexpected visualization_mode: {}", self._config.visualization_mode)
        else:

            # Update the visualizer step.
            self._global_step += 1
            for node_id in self._nodes.keys():
                tblogger = self._tbloggers[node_id]
                channel = self._channels[node_id]

                # Try to query node.
                try:
                    response = self._query_node(node_id, channel)
                except Exception as e:
                    logger.info("Failed to query node: {}", node_id)
                    continue

                # Try to log response.
                try:
                    self._log_response(node_id, response, tblogger)
                except Exception as e:
                    logger.info("Failed to log response: {}, node: {}, error: {}", response, node_id, e)
                    continue

    def _retrieve_all_nodes(self):
        # Refresh list of nodes in the network
        self._nodes = {}

        # Query the chain and retrieve all nodes that are "mining"
        payload = dict(
            code = self._config.eos.code,
            table = self._config.eos.table,
            scope = self._config.eos.scope,
            key_type = self._config.eos.key_type,
            json = "true"
        )

        payload_json = json.dumps(payload)
        request_url = self._config.eos.url + self._config.eos.get_table_command
        response = requests.post(url=request_url, data=payload_json)
        if response.status_code == HTTPStatus.OK:
            response_json = response.json()
            rows = response_json['rows']
            for row in rows:
                node_dict = dict(identity=row['identity'], url=row['address'], port=row['port'])
                self._nodes[ row['identity'] ] = node_dict
        else:
            logger.error("Error: Could not retrieve the nodes connected to the chain.")


    def _refresh_all_channels(self):
        for node_id in list(self._channels.keys()):
            if node_id not in self._nodes:
                self._channels[node_id].close()
                del self._channels[node_id]

        # Add new node channels.
        for node in self._nodes.values():
            if node['identity'] not in self._channels:
                node_url = "{}:{}".format(node['url'], node['port'])
                new_channel = grpc.insecure_channel(node_url)
                self._channels[node['identity']] = new_channel

    def _refresh_all_tbloggers(self):
        for node_id in list(self._tbloggers.keys()):
            if node_id not in self._nodes:
                del self._tbloggers[node_id]

        # Add new node loggers.
        for node_id in self._nodes.keys():
            if node_id not in self._tbloggers:
                log_dir = self._config.logdir + "/" + node_id
                self._tbloggers[node_id] = TBLogger(log_dir)

    def _query_node(self, node_id, channel):
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
        return response

    def _log_response(self, node_id, response, tblogger):
        tblogger.log_scalar("step", response['step'], self._global_step)
        tblogger.log_scalar("gs", response['gs'], self._global_step)
        tblogger.log_scalar("mem", response['mem'], self._global_step)
        tblogger.log_scalar("loss", response['loss'], self._global_step)

        # Metrics
        metrics = response['metrics']
        for idn in metrics.keys():
            tblogger.log_scalar(idn, metrics[idn], self._global_step)

        # Post scores to tb.
        scores = response['scores']
        if scores:
            for idn, score in scores:
                tblogger.log_scalar(idn, score, self._global_step)

        logger.info('Logging: node {}: step {} gs {} mem {} loss {} scores {}'.format(node_id, response['step'], response['gs'], response['mem'], response['loss'], scores))


def main(config):
    listener = NodeStatsListener(config)
    try:
        logger.info('Started listener ...')
        while True:
            listener.refresh()
            time.sleep(config.heartbeat)

    except KeyboardInterrupt:
        logger.info('Stopping listener with keyboard interrupt.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path',
        default='./visualizer/config.yaml',
        type=str,
        help='Path to config file.'
    )
    args = parser.parse_args()
    config = Config.get_config_from_yaml(args.config_path)
    main(config)
