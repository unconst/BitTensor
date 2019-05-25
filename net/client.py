import logging
import grpc
import bolt_pb2
import bolt_pb2_grpc

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = bolt_pb2_grpc.BoltStub(channel)
        thing = bolt_pb2.thing(word='cat')
        response = stub.Test(thing)
        print (response.word)


if __name__ == '__main__':
    logging.basicConfig()
    run()
