import socket
import threading
import pickle
from tetris_pc import TetrisPC

def start_server(port=5000):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("", port))
    server_socket.listen(5)
    print(f"Server listening on port {port}")

    solver = TetrisPC()

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        threading.Thread(target=handle_client, args=(client_socket, solver)).start()

def handle_client(client_socket, solver):
    data = client_socket.recv(4096)
    request = pickle.loads(data)
    suggestions = solver.suggest_moves(request['board'], request['bag'], request['hold'])
    client_socket.send(pickle.dumps(suggestions))
    client_socket.close()

if __name__ == "__main__":
    start_server()
