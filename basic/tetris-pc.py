import numpy as np
import cv2
import pyautogui
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import threading
import time
import tkinter as tk
from tkinter import ttk

class TetrisPC:
    def __init__(self):
        self.memo = {}
        self.kick_table = {
            'I': [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
            'default': [(0, 0), (-1, 0), (1, 0), (-1, 1), (1, -1)]
        }
        self.spawn_x = 4
        self.model = self.load_piece_recognition_model()
        self.auto_mode = False  # 自动模式开关

    def load_piece_recognition_model(self):
        model = SimpleCNN()
        model.load_state_dict(torch.load('tetris_piece_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model

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
                    new_board, final_x = self.hard_drop_with_kick(board, rotation, x, piece)
                    if new_board is not None:
                        if self.can_pc(new_board, new_bag, hold, depth+1):
                            self.memo[state] = True
                            return True
        
        if hold:
            for i, piece in enumerate(bag):
                if hold != piece:
                    new_bag = bag[:i] + [hold] + bag[i+1:]
                    for rotation in self.get_rotations(piece):
                        for x in range(len(board[0])):
                            new_board, final_x = self.hard_drop_with_kick(board, rotation, x, piece)
                            if new_board is not None:
                                if self.can_pc(new_board, new_bag, piece, depth+1):
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
        elif piece == 'S':
            rotations = [np.array([[0, 1, 1], [1, 1, 0]]), np.array([[1, 0], [1, 1], [0, 1]])]
        elif piece == 'Z':
            rotations = [np.array([[1, 1, 0], [0, 1, 1]]), np.array([[0, 1], [1, 1], [1, 0]])]
        elif piece == 'L':
            rotations = [np.array([[1, 0], [1, 0], [1, 1]]), np.array([[1, 1, 1], [1, 0, 0]]),
                         np.array([[1, 1], [0, 1], [0, 1]]), np.array([[0, 0, 1], [1, 1, 1]])]
        elif piece == 'J':
            rotations = [np.array([[0, 1], [0, 1], [1, 1]]), np.array([[1, 0, 0], [1, 1, 1]]),
                         np.array([[1, 1], [1, 0], [1, 0]]), np.array([[1, 1, 1], [0, 0, 1]])]
        return rotations

    def perform_moves(self, moves):
        if self.auto_mode:
            for move in moves:
                if 'left' in move:
                    pyautogui.press('left', presses=abs(int(move.split()[2])))
                elif 'right' in move:
                    pyautogui.press('right', presses=int(move.split()[2]))
                if 'HD' in move:
                    pyautogui.press('space')
        else:
            for move in moves:
                print(move)

    def display_move_instructions(self, piece, target_column):
        moves = target_column - self.spawn_x
        instruction = ""
        if moves > 0:
            instruction = f"Move {piece} right {moves} times, then HD (Hard Drop)."
        elif moves < 0:
            instruction = f"Move {piece} left {abs(moves)} times, then HD (Hard Drop)."
        else:
            instruction = f"HD (Hard Drop) {piece} from spawn point."
        self.perform_moves([instruction])

    def gui(self):
        root = tk.Tk()
        root.title("Tetris PC Assistant")
        
        mode_label = ttk.Label(root, text="Select Mode:")
        mode_label.pack(pady=10)

        auto_button = ttk.Button(root, text="Auto Mode", command=self.enable_auto)
        auto_button.pack(pady=5)

        suggest_button = ttk.Button(root, text="Suggest Only", command=self.disable_auto)
        suggest_button.pack(pady=5)

        root.mainloop()

    def enable_auto(self):
        self.auto_mode = True
        print("Auto mode enabled. The system will perform moves automatically.")

    def disable_auto(self):
        self.auto_mode = False
        print("Suggest mode enabled. The system will only display move instructions.")

if __name__ == '__main__':
    tetris_bot = TetrisPC()
    tetris_bot.gui()
