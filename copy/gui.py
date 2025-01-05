import tkinter as tk
from tkinter import Canvas, Label, Button
import numpy as np
import pyautogui
from tetris.tetris_pc import TetrisPC
from tetris.screen_capture import capture_screen, find_tetris_board, extract_board_state
from tetris.piece_classifier import identify_next_piece
import time

class TetrisGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Tetris PC Helper")

        # 创建界面布局
        self.canvas = Canvas(master, width=300, height=600, bg='black')
        self.canvas.grid(row=0, column=0, rowspan=4)

        self.label_status = Label(master, text="Status: Monitoring", font=("Helvetica", 14))
        self.label_status.grid(row=0, column=1, padx=10, pady=10)
        
        self.start_button = Button(master, text="Start PC", command=self.start_pc, font=("Helvetica", 12))
        self.start_button.grid(row=1, column=1, pady=10)

        self.stop_button = Button(master, text="Stop", command=self.stop_pc, font=("Helvetica", 12))
        self.stop_button.grid(row=2, column=1, pady=10)

        self.exit_button = Button(master, text="Exit", command=master.quit, font=("Helvetica", 12))
        self.exit_button.grid(row=3, column=1, pady=10)

        self.tetris_bot = TetrisPC()
        self.running = True
        self.current_piece = None
        self.current_piece_position = [4, 0]  # 初始位置

        self.update_screen()

    def draw_board(self, board):
        cell_size = 30
        self.canvas.delete("all")
        for y, row in enumerate(board):
            for x, cell in enumerate(row):
                if cell > 0:
                    self.canvas.create_rectangle(
                        x * cell_size, y * cell_size,
                        (x + 1) * cell_size, (y + 1) * cell_size,
                        fill="blue", outline="gray"
                    )

    def animate_piece(self, path):
        cell_size = 30
        if path:
            next_move = path.pop(0)
            if 'left' in next_move:
                self.current_piece_position[0] -= 1
            elif 'right' in next_move:
                self.current_piece_position[0] += 1
            elif 'HD' in next_move:
                self.current_piece_position[1] += 1

            self.canvas.create_rectangle(
                self.current_piece_position[0] * cell_size,
                self.current_piece_position[1] * cell_size,
                (self.current_piece_position[0] + 1) * cell_size,
                (self.current_piece_position[1] + 1) * cell_size,
                fill="yellow", outline="gray"
            )
            self.master.after(200, lambda: self.animate_piece(path))  # 每200ms移动一次

    def update_screen(self):
        if self.running:
            screen = capture_screen()
            board_region = find_tetris_board(screen)
            if board_region:
                board_state = extract_board_state(screen, board_region)
                self.draw_board(board_state)
            self.master.after(100, self.update_screen)

    def start_pc(self):
        self.label_status.config(text="Status: PC Running")
        screen = capture_screen()
        board_region = find_tetris_board(screen)

        if board_region:
            board_state = extract_board_state(screen, board_region)
            next_piece = identify_next_piece(screen, (board_region[0] + 250, board_region[1] - 100), self.tetris_bot.model)
            
            moves = self.tetris_bot.real_time_suggest(board_state, [next_piece], None)
            self.animate_piece(moves)

    def stop_pc(self):
        self.label_status.config(text="Status: Stopped")
        self.running = False

if __name__ == '__main__':
    root = tk.Tk()
    gui = TetrisGUI(root)
    root.mainloop()
