import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Canvas
import socket
import threading
import pickle

class TetrisPC:
    def __init__(self):
        self.memo = {}

    def can_pc(self, board, bag, hold, depth=0):
        state = (tuple(map(tuple, board)), tuple(bag), hold, depth)
        
        if state in self.memo:
            return self.memo[state]
        
        if self.is_perfect_clear(board):
            return True
        
        if depth == len(bag):
            return False

        for i, piece in enumerate(bag):
            new_bag = bag[:i] + bag[i+1:]
            for rotation in self.get_rotations(piece):
                for x in range(len(board[0])):
                    new_board = self.place_piece(board, rotation, x)
                    if new_board is not None:
                        if self.can_pc(new_board, new_bag, hold, depth+1):
                            self.memo[state] = True
                            return True

        if hold:
            for rotation in self.get_rotations(hold):
                for x in range(len(board[0])):
                    new_board = self.place_piece(board, rotation, x)
                    if new_board is not None:
                        if self.can_pc(new_board, bag, hold, depth+1):
                            self.memo[state] = True
                            return True

        self.memo[state] = False
        return False

    def is_perfect_clear(self, board):
        return np.all(board == 0)

    def get_rotations(self, piece):
        rotations = []
        if piece == 'I':
            rotations = [np.array([[1, 1, 1, 1]]), np.array([[1], [1], [1], [1]])]
        elif piece == 'O':
            rotations = [np.array([[1, 1], [1, 1]])]
        elif piece == 'T':
            rotations = [np.array([[1, 1, 1], [0, 1, 0]]), np.array([[1, 0], [1, 1], [1, 0]]),
                         np.array([[0, 1, 0], [1, 1, 1]]), np.array([[0, 1], [1, 1], [0, 1]])]
        return rotations

    def place_piece(self, board, piece, x):
        new_board = board.copy()
        y = 0
        while y + piece.shape[0] <= new_board.shape[0]:
            if np.any(new_board[y:y+piece.shape[0], x:x+piece.shape[1]] + piece > 1):
                break
            y += 1
        y -= 1

        if y >= 0:
            new_board[y:y+piece.shape[0], x:x+piece.shape[1]] += piece
            return new_board
        return None

    def suggest_moves(self, board, bag, hold):
        suggestions = []
        for i, piece in enumerate(bag):
            new_bag = bag[:i] + bag[i+1:]
            for rotation in self.get_rotations(piece):
                for x in range(len(board[0])):
                    new_board = self.place_piece(board, rotation, x)
                    if new_board is not None and self.can_pc(new_board, new_bag, hold):
                        suggestions.append((piece, rotation, x, new_board))
        return suggestions

    def plot_board(self, board):
        plt.imshow(board, cmap='gray_r')
        plt.xticks(np.arange(-.5, 10, 1), [])
        plt.yticks(np.arange(-.5, 20, 1), [])
        plt.grid(True, which='both', color='black', linewidth=0.5)
        plt.title('Tetris Board')
        plt.show()

    def launch_gui(self, suggestions):
        root = tk.Tk()
        root.title("Tetris PC Suggestions")
        canvas = Canvas(root, width=300, height=600)
        canvas.pack()
        
        for _, _, _, new_board in suggestions:
            self.draw_board(canvas, new_board)
            root.update_idletasks()
            root.update()
            canvas.delete("all")

        root.mainloop()

    def draw_board(self, canvas, board):
        cell_size = 30
        for y, row in enumerate(board):
            for x, cell in enumerate(row):
                if cell > 0:
                    canvas.create_rectangle(
                        x * cell_size, y * cell_size,
                        (x + 1) * cell_size, (y + 1) * cell_size,
                        fill="blue"
                    )

    def start_server(self, port=5000):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(("", port))
        server_socket.listen(5)
        print(f"Server listening on port {port}")

        while True:
            client_socket, addr = server_socket.accept()
            print(f"Connection from {addr}")
            threading.Thread(target=self.handle_client, args=(client_socket,)).start()

    def handle_client(self, client_socket):
        data = client_socket.recv(4096)
        request = pickle.loads(data)
        suggestions = self.suggest_moves(request['board'], request['bag'], request['hold'])
        client_socket.send(pickle.dumps(suggestions))
        client_socket.close()

# Example usage
if __name__ == "__main__":
    board = np.zeros((20, 10), dtype=int)
    bag = ['T', 'O', 'I', 'S', 'Z', 'L', 'J']  # 7-bag system
    hold = None

    solver = TetrisPC()
    threading.Thread(target=solver.start_server, daemon=True).start()
    suggestions = solver.suggest_moves(board, bag, hold)

    if suggestions:
        solver.launch_gui(suggestions)
