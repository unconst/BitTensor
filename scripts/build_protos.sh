# Build protos.
cd "$(dirname "$0")"/..
python -m grpc.tools.protoc bittensor/proto/bolt.proto  -I. --python_out=. --grpc_python_out=.
