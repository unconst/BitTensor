#!/bin/bash
import argparse
import grpc
from concurrent import futures

import visualizer

class NodeStatsListener():
    def __init__(self, args):
        self.args = args
        # Init server.
        self.server_address = self.args.bind_address + ":" + self.args.port
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        visualizer.proto.visualizer_pb2_grpc.add_VisualizerServicer_to_server(
            self, self.server)
        self.server.add_insecure_port(self.server_address)
    
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
   
    

    