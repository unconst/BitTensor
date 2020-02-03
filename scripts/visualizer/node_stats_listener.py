#!/bin/bash
import argparse
import grpc
import subprocess
import visualizer
import requests
import json
import pickle
import time


from config import Config
from concurrent import futures
from http import HTTPStatus
from loguru import logger
from TBLogger import TBLogger

class NodeStatsListener():
    
    def __init__(self, args):
        # Init configurations and arguments
        self.config = Config()

        self.args = args
        
        # Init server.
        self.server_address = self.args.bind_address + ":" + self.args.port
        
        # init node list
        self.node_list = []
        
        # init command list
        self.get_tb_metrics = "tb_metrics"
        self._is_listening = True
        
    
    
    def retrieve_all_nodes(self):
        # Query the chain and retrieve all nodes that are "mining"
        
        payload = dict(
            code = self.config.eos_code, 
            table = self.config.eos_table,
            scope = self.config.eos_scope,
            key_type = self.config.eos_key_type,
            json = "true"
        )
        
        # reset node list
        self.node_list = []
        payload_json = json.dumps(payload)
        try:
            response = requests.post(url = self.config.eos_get_table_rows, data = payload_json)
            
            if response.status_code == HTTPStatus.OK:
                response_json = response.json()
                rows = response_json['rows']
                
                for row in rows:
                    node = dict(identity=row['identity'], url=row['address'], port=row['port'])
                    
                    # TODO: If node list already contains this node, don't append it.
                    self.node_list.append(node)
            else:
                logger.error("Error: Could not retrieve the nodes connected to the chain.")
        except Exception as e:
            logger.exception("Unexpected exception. Likely an issue connecting to the EOS chain", e)
        
        logger.info("Found {} nodes.".format(len(self.node_list)))

    
    def _run(self):
        if self.config.visualization_mode == "tensorboard":
            logger.info("Tensorboard mode, listening on channels and collecting data to logdirs")
            self.collect_data_to_tensorboard()
    
        
    def collect_data_to_tensorboard(self):
        request_payload_bytes = pickle.dumps(self.get_tb_metrics, protocol=0)

        while self._is_listening:
            self.retrieve_all_nodes()
            
            for node in self.node_list:
                identity_dir = node['identity']
                log_dir = self.args.logdir + "/" + identity_dir
                _tblogger = TBLogger(log_dir)
                
                node_url = "{}:{}".format(node['url'], node['port'])
                                
                # open up channels for running nodes
                self.channel = grpc.insecure_channel(node_url)
                self.stub = visualizer.visualizer_pb2_grpc.VisualizerStub(self.channel)
                
                response = self.stub.Report(
                    visualizer.visualizer_pb2.ReportRequest(
                        version = 1.0,
                        source_id = '2',
                        payload = request_payload_bytes
                    )
                )
                # Close the channel since we're done here.
                self.channel.close()
                
                # Let's process the incoming data
                payload = pickle.loads(response.payload)
                step = payload['step']
                _tblogger.log_scalar("gs", payload['gs'], step)
                
                # Log memory size.
                _tblogger.log_scalar("mem", payload['mem'], step)

                # Log loss
                _tblogger.log_scalar("loss", payload['loss'], step)

                # Metrics
                metrics = payload['metrics']
                for idn in metrics.keys():
                    _tblogger.log_scalar(idn, metrics[idn], step)
                
                # Post scores to tb.
                scores = payload['scores']
                if scores:
                    for idn, score in scores:
                        _tblogger.log_scalar(idn, score, step)
                
                logger.info('node {}: gs {} mem {} loss {} scores {}'.format(identity_dir, payload['gs'], payload['mem'], payload['loss'], scores))
                
                time.sleep(10)
            
            
    def __del__(self):
        self._stop_listening()
        logger.debug('Stopped Listening at: {}.', self.server_address)
    
    
    def _start_listening(self):
        self._is_listening = True
        self._run()
        
        
    def _stop_listening(self):
        self._is_listening = False
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("bind_address", help="Address to which to bind the node listener gRPC server")
    parser.add_argument("port", help="Port that the stats listener will be listening on")
    parser.add_argument("logdir", help="Directory to which all tensorboard log files will be written.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    listener = NodeStatsListener(args)
    listener._start_listening()
    
    
   


    