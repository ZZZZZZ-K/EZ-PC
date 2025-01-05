from tetris_pc import TetrisPC
import threading
import numpy as np

if __name__ == "__main__":
    board = np.zeros((20, 10), dtype=int)
    bag = ['T', 'O', 'I', 'S', 'Z', 'L', 'J']
    hold = None

    solver = TetrisPC()
    threading.Thread(target=solver.start_server, daemon=True).start()
    suggestions = solver.suggest_moves(board, bag, hold)

    if suggestions:
        solver.launch_gui(suggestions)
