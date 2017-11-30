import zmq

port = "5001"

context = zmq.Context()
socket = context.socket(zmq.REP)

socket.bind(f"tcp://*:{port}")

message = socket.recv()
print(message)

answer = {
    'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
}

socket.send_json(answer)

print("Finish him!!")
