# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from visualizer.proto import visualizer_pb2 as visualizer__pb2


class VisualizerStub(object):
  """Visualizer protocol to define messages passed 
  between nodes and visualizer server.

  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Report = channel.unary_unary(
        '/Visualizer/Report',
        request_serializer=visualizer__pb2.ReportRequest.SerializeToString,
        response_deserializer=visualizer__pb2.ReportResponse.FromString,
        )


class VisualizerServicer(object):
  """Visualizer protocol to define messages passed 
  between nodes and visualizer server.

  """

  def Report(self, request, context):
    """Query remote node for a report on its findings thus far, response is an object containing 
    loss, gs, step, mem, etc.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_VisualizerServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Report': grpc.unary_unary_rpc_method_handler(
          servicer.Report,
          request_deserializer=visualizer__pb2.ReportRequest.FromString,
          response_serializer=visualizer__pb2.ReportResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'Visualizer', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))