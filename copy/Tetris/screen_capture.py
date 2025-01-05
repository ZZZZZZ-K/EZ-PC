import cv2
import numpy as np
import pyautogui

def capture_screen(region=None):
    screenshot = pyautogui.screenshot(region=region)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

def find_tetris_board(screen):
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 200 < w < 400 and 400 < h < 800:
            return (x, y, w, h)
    return None

def extract_board_state(frame, board_region):
    x, y, w, h = board_region
    board = frame[y:y+h, x:x+w]
    cell_size = w // 10
    state = np.zeros((20, 10), dtype=int)

    for row in range(20):
        for col in range(10):
            cell = board[row * cell_size:(row+1) * cell_size, col * cell_size:(col+1) * cell_size]
            if np.mean(cell) < 200:
                state[row, col] = 1
    return state
