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
        self.kick_table = {
            'I': [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
            'default': [(0, 0), (-1, 0), (1, 0), (-1, 1), (1, -1)]
        }
        self.spawn_x = 4  # 通常的方块生成位置

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
                    new_board = self.hard_drop(board, rotation, x, piece)
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
                            new_board = self.hard_drop(board, rotation, x, piece)
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

    def hard_drop(self, board, piece, x, piece_type):
        new_board = board.copy()
        y = 0
        while y + piece.shape[0] <= new_board.shape[0]:
            if np.any(new_board[y:y+piece.shape[0], x:x+piece.shape[1]] + piece > 1):
                break
            y += 1
        y -= 1

        if y >= 0:
            new_board[y:y+piece.shape[0], x:x+piece.shape[1]] += piece
            self.display_move_instructions(piece, x)
            return new_board
        return None

    def display_move_instructions(self, piece, target_column):
        moves = target_column - self.spawn_x
        if moves > 0:
            print(f"Move {piece} right {moves} times, then HD (Hard Drop).")
        elif moves < 0:
            print(f"Move {piece} left {abs(moves)} times, then HD (Hard Drop).")
        else:
            print(f"HD (Hard Drop) {piece} from spawn point.")

    def suggest_moves(self, board, bag, hold):
        suggestions = []
        for i, piece in enumerate(bag):
            new_bag = bag[:i] + bag[i+1:]
            for rotation in self.get_rotations(piece):
                for x in range(len(board[0])):
                    new_board = self.hard_drop(board, rotation, x, piece)
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
