#!/bin/bash
import argparse
import grpc
import subprocess
import visualizer
import requests
import json

from config import Config
from concurrent import futures
from http import HTTPStatus

class NodeStatsListener():
    
    def __init__(self, args):
        # Init configurations and arguments
        self.config = Config()
        self.args = args
        
        # Init server.
        self.server_address = self.args.bind_address + ":" + self.args.port
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        visualizer.proto.visualizer_pb2_grpc.add_VisualizerServicer_to_server(
            self, self.server)
        self.server.add_insecure_port(self.server_address)
        
        # init node list
        self.node_list = []
    
    def retrieve_all_nodes(self):
        # Query the chain and retrieve all nodes that are "mining"
        
        payload = dict(
            code = self.config.eos_code, 
            table = self.config.eos_table,
            scope = self.config.eos_scope,
            key_type = self.config.eos_key_type,
            json = "true"
            )
        
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
                print("Error: Could not retrieve the nodes connected to the chain.")
        except Exception as e:
            raise "Unexpected exception. Likely an issue connecting to the EOS chain"
        
        print("Found {} nodes.".format(len(self.node_list)))
    
    def Report(self, request, context):
        raise NotImplementedError

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("bind_address", help="Address to which to bind the node listener gRPC server")
    parser.add_argument("port", help="Port that the stats listener will be listening on")
    parser.add_argument("logdir", help="Directory to which all tensorboard log files will be written.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    nsl = NodeStatsListener(args)
    nsl.retrieve_all_nodes()
   
    

    