import numpy as np
import torch
import pyautogui


class TetrisPC:
    def __init__(self):
        self.memo = {}
        self.kick_table = {
            'I': [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
            'default': [(0, 0), (-1, 0), (1, 0), (-1, 1), (1, -1)]
        }
        self.spawn_x = 4
        self.model = self.load_piece_recognition_model()

    def load_piece_recognition_model(self):
        model = SimpleCNN()
        model.load_state_dict(torch.load('model/tetris_piece_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model

    def real_time_suggest(self, board, bag, hold):
        # 动态规划PC路径，考虑未来2-3个Piece
        best_moves = []
        max_depth = 3  # 规划深度，考虑未来3个方块

        def dfs(current_board, bag, hold, depth):
            if depth == max_depth or len(bag) == 0:
                return 0

            best_score = -1
            best_sequence = []

            # 遍历所有方块以及Hold情况
            for i, piece in enumerate(bag):
                next_bag = bag[:i] + bag[i+1:]
                for rotation in self.get_rotations(piece):
                    for x in range(len(board[0])):
                        new_board, valid = self.place_piece(current_board, rotation, x, piece)
                        if valid:
                            score = self.evaluate_board(new_board)
                            future_score = dfs(new_board, next_bag, hold, depth + 1)
                            total_score = score + future_score

                            if total_score > best_score:
                                best_score = total_score
                                best_sequence = [f"Move {piece} to {x}, then HD"]

            # 尝试Hold
            if hold and hold != bag[0]:
                hold_bag = [hold] + bag[1:]
                hold_sequence = dfs(board, hold_bag, bag[0], depth)
                if hold_sequence > best_score:
                    best_sequence = [f"Hold {bag[0]}"] + hold_sequence

            if depth == 0:
                return best_sequence
            return best_score

        # 调用深度搜索，获得最优路径
        best_moves = dfs(board, bag, hold, 0)
        return best_moves

    def place_piece(self, board, piece, x, piece_type):
        # 硬掉逻辑和踢墙优化
        y = 0
        while y + piece.shape[0] <= board.shape[0]:
            if np.any(board[y:y + piece.shape[0], x:x + piece.shape[1]] + piece > 1):
                break
            y += 1
        y -= 1

        if y >= 0:
            board[y:y + piece.shape[0], x:x + piece.shape[1]] += piece
            return board, True
        return board, False

    def evaluate_board(self, board):
        # 简单评估函数：计算空洞数量和版面高度
        holes = np.sum((board == 0) & (np.cumsum(board, axis=0) > 0))
        height = np.max(np.sum(board > 0, axis=0))
        return -holes - height * 5  # 负号代表希望最小化空洞和高度

    def get_rotations(self, piece):
        rotations = []
        if piece == 'I':
            rotations = [np.array([[1, 1, 1, 1]]), np.array([[1], [1], [1], [1]])]
        elif piece == 'O':
            rotations = [np.array([[1, 1], [1, 1]])]
        elif piece == 'T':
            rotations = [
                np.array([[1, 1, 1], [0, 1, 0]]),
                np.array([[1, 0], [1, 1], [1, 0]]),
                np.array([[0, 1, 0], [1, 1, 1]]),
                np.array([[0, 1], [1, 1], [0, 1]])
            ]
        return rotations

    def perform_moves(self, moves):
        for move in moves:
            if 'left' in move:
                pyautogui.press('left', presses=int(move.split()[2]))
            elif 'right' in move:
                pyautogui.press('right', presses=int(move.split()[2]))
            if 'HD' in move:
                pyautogui.press('space')
            if 'Hold' in move:
                pyautogui.press('shift')  # Hold键通常绑定到C
