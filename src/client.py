import logging
import grpc
import proto.bolt_pb2
import proto.bolt_pb2_grpc

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = proto.bolt_pb2_grpc.BoltStub(channel)
        words = proto.bolt_pb2.Words(words=['cat'])
        response = stub.Spike(words)
        print (response)


if __name__ == '__main__':
    logging.basicConfig()
    run()
