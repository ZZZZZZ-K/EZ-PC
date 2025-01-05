# EZ PC

This project is an automated Perfect Clear (PC) solver for Tetris, designed to recognize the current game board and predict the best move sequence using deep learning and dynamic programming (DP) algorithms.

## Features
- Real-time Tetris board detection using screen capture
- CNN model to recognize next Tetris pieces (I, O, T, S, Z, L, J)
- Automated PC path planning and execution
- Animated GUI to visualize piece movement and board state

## Project Structure
```
Tetris-Bot/
│
├── main.py                     # Main entry to start the GUI and PC solver
├── requirements.txt            # List of dependencies (to be created)
├── gui.py                      # GUI interface for visualization
├── model/
│   └── tetris_piece_model.pth  # Pre-trained model for Tetris piece recognition
├── tetris/
│   ├── __init__.py             # Package initialization
│   ├── tetris_pc.py            # PC solver logic
│   ├── screen_capture.py       # Screen capturing and board detection
│   └── piece_classifier.py     # CNN model for piece classification and training
└── data/                       # Dataset for training and testing (optional)
```

## Prerequisites
- Python 3.8 or higher
- CUDA (if available) for GPU acceleration

## Installation
1. **Install Python**:
   Download and install Python from [python.org](https://www.python.org/downloads/).

2. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Tetris-Bot.git
   cd Tetris-Bot
   ```

3. **Install dependencies**:
   (Create a `requirements.txt` with necessary packages)
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Model**:
   Place the pre-trained model `tetris_piece_model.pth` in the `model/` directory. If not available, train it using `piece_classifier.py`.

## Usage
1. **Run the GUI**:
   ```bash
   python main.py
   ```
   This will start the Tetris PC Solver interface. Use the GUI to monitor the board and automate moves.

2. **Train Model (Optional)**:
   ```bash
   python tetris/piece_classifier.py
   ```
   Train your own model using a custom dataset located in `data/train` and `data/test`.

## Notes
- Ensure your Tetris game is in a visible area on the screen for the solver to detect.
- This solver is designed for 7-bag randomizers and may not perform well with non-standard Tetris versions.

---

Contributions are welcome! Feel free to open issues or submit PRs for improvements.
