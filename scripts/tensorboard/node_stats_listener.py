#!/bin/bash
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("port", help="Port that the stats listener will be listening on")
    parser.add_argument("logdir", help="Directory to which all tensorboard log files will be written.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    print(args.logdir)
    print(args.port)

    