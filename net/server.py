from concurrent import futures
import logging
import time
import grpc
import bolt_pb2
import bolt_pb2_grpc

_ONE_DAY_IN_SECONDS = 60*60*24

class BoltServicer(bolt_pb2_grpc.BoltServicer):
    def __init__(self):
        pass

    def Test(self, request, context):
        print (request.word)
        return bolt_pb2.thing(word=request.word)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    bolt_pb2_grpc.add_BoltServicer_to_server(
        BoltServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig()
    serve()

