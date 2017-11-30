import zmq

address = [
    "127.0.0.1:5001", "127.0.0.1:5002", "127.0.0.1:5003", "127.0.0.1:5004",
    "127.0.0.1:5005"
]

param_grid = {
    'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'dropout_rate': [0.2, 0.3, 0.4, 0.5],
    'units': [4, 8, 16, 32]
}

answers = {}
context = zmq.Context()
socket = context.socket(zmq.REQ)

for i in range(len(address)-1):
    socket.connect(f"tcp://{address[i]}")
    message = {
        'learning_rate': [param_grid['learning_rate'][i]],
        'dropout_rate': param_grid['dropout_rate'],
        'units': param_grid['units']
    }

    socket.send_json(message)
    answers[address[i]] = socket.recv_json()
    print("Finish him!! of: ", address[i])
