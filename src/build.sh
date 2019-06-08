# Build protos.
cd "$(dirname "$0")"
python -m grpc_tools.protoc proto/tensorflow/core/framework/*.proto  -I. --python_out=. --grpc_python_out=.
python -m grpc_tools.protoc proto/bolt.proto  -I. --python_out=. --grpc_python_out=.
