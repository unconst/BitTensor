#!/bin/bash
import argparse
import grpc
from concurrent import futures

import visualizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("bind_address", help="Address to which to bind the node listener gRPC server")
    parser.add_argument("port", help="Port that the stats listener will be listening on")
    parser.add_argument("logdir", help="Directory to which all tensorboard log files will be written.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # Init server.
    server_address = args.bind_address + ":" + args.port
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    visualizer.proto.visualizer_pb2_grpc.add_BittensorServicer_to_server(
        self, server)
    server.add_insecure_port(server_address)
    

    