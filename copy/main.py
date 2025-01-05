import tkinter as tk
from tetris.tetris_pc import TetrisPC
from tetris.screen_capture import capture_screen, find_tetris_board, extract_board_state
from tetris.piece_classifier import identify_next_piece
from gui import TetrisGUI
import time


def main():
    # 创建主窗口，加载Tetris GUI
    root = tk.Tk()
    gui = TetrisGUI(root)
    root.mainloop()

    # 初始化自动PC算法
    tetris_bot = TetrisPC()

    while True:
        screen = capture_screen()
        board_region = find_tetris_board(screen)
        
        if board_region:
            # 获取当前版面状态
            board_state = extract_board_state(screen, board_region)
            
            # 识别下一个Piece
            next_piece = identify_next_piece(
                screen, (board_region[0] + 250, board_region[1] - 100), tetris_bot.model
            )
            
            # 计算最优路径并执行
            moves = tetris_bot.real_time_suggest(board_state, [next_piece], None)
            tetris_bot.perform_moves(moves)

        time.sleep(0.1)  # 100ms监测一次，避免占用过多CPU


if __name__ == '__main__':
    main()
