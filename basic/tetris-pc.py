import numpy as np
import cv2
import pyautogui
import torch
import torch.nn as nn
import torchvision.transforms as T

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
        # 加载预训练模型或初始化一个简单的CNN模型
        model = SimpleCNN()
        model.load_state_dict(torch.load('tetris_piece_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model

    def capture_screen(self, region=None):
        screenshot = pyautogui.screenshot(region=region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def find_tetris_board(self, screen):
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 200 < w < 400 and 400 < h < 800:
                return (x, y, w, h)
        return None

    def extract_board_state(self, frame, board_region):
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

    def identify_next_piece(self, frame, queue_region):
        next_piece_area = frame[queue_region[1]:queue_region[1]+100, queue_region[0]:queue_region[0]+100]
        transform = T.Compose([T.ToTensor(), T.Resize((28, 28))])
        piece_tensor = transform(next_piece_area).unsqueeze(0)
        output = self.model(piece_tensor)
        _, predicted = torch.max(output, 1)
        pieces = ['I', 'O', 'T', 'S', 'Z', 'L', 'J']
        return pieces[predicted.item()]

    def perform_moves(self, moves):
        for move in moves:
            if 'left' in move:
                pyautogui.press('left', presses=abs(int(move.split()[2])))
            elif 'right' in move:
                pyautogui.press('right', presses=int(move.split()[2]))
            if 'HD' in move:
                pyautogui.press('space')

    def main_loop(self):
        while True:
            screen = self.capture_screen()
            board_region = self.find_tetris_board(screen)
            
            if board_region:
                board_state = self.extract_board_state(screen, board_region)
                next_piece = self.identify_next_piece(screen, (board_region[0] + 250, board_region[1] - 100))
                
                moves = self.real_time_suggest(board_state, [next_piece], None)
                self.perform_moves(moves)
            
            pyautogui.sleep(0.1)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    tetris_bot = TetrisPC()
    tetris_bot.main_loop()
