from copy import copy, deepcopy
import numpy as np
from piece import BODIES, Piece
from board import Board
from random import randint
import random
import pickle

def peaks(board):
        
    peaks = np.array([])
    arow, acol = board.shape[1], board.shape[0]
        
    for col in range(arow):
        if 1 in board[:, col]:
            x = acol - np.argmax(board[:, col][::-1], axis=0) # finds the distance from the bottom of the board to the first 
            peaks = np.append(peaks, x)
        else:
            peaks = np.append(peaks, 0)
                
    return peaks

def bumpiness(apeaks):
        
    total = 0
        
    for i in range(9):
        total += np.abs(apeaks[i] - apeaks[i + 1])
            
    return total


class CUSTOM_AI_MODEL:
    def __init__(self, weights = None, afeatures = 3, mutate = False, noise = 0.1): 
        self.weights = weights
        self.afeatures = afeatures
        self.mutate = mutate
        self.noise = noise

        if self.weights is None:
            self.weights = np.array([-1.19436948, -0.19174679, -0.27143913])
        elif mutate == False:
            self.weights = weights
        else:
            self.weights = weights * (np.array([np.random.normal(1, noise) for i in range(afeatures)])) # adds additional randomness at each generation

        self.fit_score = 0.0 # individual fitness
        self.fit_rel = 0.0 # relative fitness compared to other agents

    def __lt__(self, other):
        return (self.fit_score < other.fit_score)

    def eval(self, board):
        peak = peaks(board)
        bump = bumpiness(peak)

        valscore = np.array([np.sum(peak), bump, np.count_nonzero(np.mean(board, axis=1))])

        return np.dot(valscore, self.weights)


    def get_best_move(self, board, piece):
        best_x = -1000
        max_value = -1000
        best_piece = None
        
        for i in range(4):
            piece = piece.get_next_rotation()
            for x in range(board.width):
                try:
                    y = board.drop_height(piece, x)
                except:
                    continue

                board_copy = deepcopy(board.board)
                for pos in piece.body:
                    board_copy[y + pos[1]][x + pos[0]] = True

                f = lambda x: 1 if x == True else 0

                board_copy = np.asarray([[f(j) for j in i] for i in board_copy])
                c = self.eval(board_copy)

                if c > max_value:
                    max_value = c
                    best_x = x
                    best_piece = piece
                    
        return best_x, best_piece
    
