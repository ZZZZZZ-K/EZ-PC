import torch
import torch.nn as nn
import torchvision.transforms as T

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 7)  # 7类方块

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def identify_next_piece(frame, queue_region, model):
    next_piece_area = frame[queue_region[1]:queue_region[1]+100, queue_region[0]:queue_region[0]+100]
    transform = T.Compose([T.ToTensor(), T.Resize((28, 28))])
    piece_tensor = transform(next_piece_area).unsqueeze(0)
    output = model(piece_tensor)
    _, predicted = torch.max(output, 1)
    pieces = ['I', 'O', 'T', 'S', 'Z', 'L', 'J']
    return pieces[predicted.item()]
