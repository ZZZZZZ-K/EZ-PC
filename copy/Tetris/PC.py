import torch
import pyautogui
from tetris.tetris_pc import TetrisPC
from tetris.screen_capture import capture_screen, find_tetris_board, extract_board_state
from tetris.piece_classifier import identify_next_piece
import time

def main():
    # 初始化俄罗斯方块PC解决方案类
    tetris_bot = TetrisPC()

    # 配置区域捕捉的初始参数
    queue_offset_x = 250
    queue_offset_y = -100

    print("Tetris bot is running... Press Ctrl+C to stop.")
    try:
        while True:
            screen = capture_screen()
            board_region = find_tetris_board(screen)

            if board_region:
                # 提取当前版面状态
                board_state = extract_board_state(screen, board_region)
                # 动态计算下一个Piece的区域
                queue_region = (board_region[0] + queue_offset_x, board_region[1] + queue_offset_y)
                # 识别下一个Piece
                next_piece = identify_next_piece(screen, queue_region, tetris_bot.model)
                
                # 计算推荐移动路径
                moves = tetris_bot.real_time_suggest(board_state, [next_piece], None)
                # 执行推荐操作
                tetris_bot.perform_moves(moves)
            else:
                print("Tetris board not found. Ensure the game is running and visible.")

            time.sleep(0.1)  # 休眠0.1秒，减少CPU占用
    
    except KeyboardInterrupt:
        print("Tetris bot stopped.")

if __name__ == '__main__':
    main()
