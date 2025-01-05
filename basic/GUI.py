import tkinter as tk
from tkinter import Canvas

class TetrisGUI:
    def __init__(self, suggestions):
        self.suggestions = suggestions
        self.root = tk.Tk()
        self.root.title("Tetris PC Suggestions")
        self.canvas = Canvas(self.root, width=300, height=600)
        self.canvas.pack()

    def draw_board(self, board):
        cell_size = 30
        self.canvas.delete("all")
        for y, row in enumerate(board):
            for x, cell in enumerate(row):
                if cell > 0:
                    self.canvas.create_rectangle(
                        x * cell_size, y * cell_size,
                        (x + 1) * cell_size, (y + 1) * cell_size,
                        fill="blue"
                    )

    def run(self):
        for _, _, _, new_board in self.suggestions:
            self.draw_board(new_board)
            self.root.update_idletasks()
            self.root.update()
        self.root.mainloop()
