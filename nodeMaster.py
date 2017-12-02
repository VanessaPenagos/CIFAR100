import zmq

address = ["192.168.1.200:5000", "192.168.1.225:5000"]

param_grid = {
    'learning_rate': [1e-3, 1e-2],
    'dropout_rate': [0.2, 0.4, 0.8],
    'units': [4, 8, 16]
}

answers = {}
context = zmq.Context()
socket = context.socket(zmq.REQ)

for i in range(len(address)):
    socket.connect(f"tcp://{address[i]}")
    message = {
        'learning_rate': [param_grid['learning_rate']],
        'dropout_rate': [param_grid['dropout_rate'][i]],
        'units': [param_grid['units']]
    }

    socket.send_json(message)
    answers[address[i]] = socket.recv_json()
    print("Finish him!! of: ", address[i])

    best_score = max(answers.keys())
    print(f"The best score is: {best_score}")
