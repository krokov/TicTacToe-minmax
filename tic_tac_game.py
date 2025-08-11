import numpy as np
import random
import pickle
import os


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1 # 1 for X, -1 for O
        self.game_over = False
        self.winner = 0 # 0: no winner, 1: player X wins, -1: player O wins, 2: draw

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = 0
        return tuple(self.board.flatten())

    def get_available_moves(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r, c] == 0]

    def make_move(self, action):
        if action not in self.get_available_moves():
            return tuple(self.board.flatten()), -100, True, "Invalid move"

        self.board[action[0], action[1]] = self.current_player

        reward = 0
        done = False
        status = "Valid move"

        if self._check_win():
            self.game_over = True
            self.winner = self.current_player
            reward = 10
            done = True
        elif len(self.get_available_moves()) == 0:
            self.game_over = True
            self.winner = 2
            reward = 1
            done = True
        else:
            self.current_player *= -1

        return tuple(self.board.flatten()), reward, done, status

    def _check_win(self):
        for player_val in [1, -1]:
            for row in range(3):
                if np.all(self.board[row, :] == player_val): return True
            for col in range(3):
                if np.all(self.board[:, col] == player_val): return True
            if np.all(np.diag(self.board) == player_val) or \
               np.all(np.diag(np.fliplr(self.board)) == player_val): return True
        return False

    def render(self):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        print("-------------")
        for row in range(3):
            print("|", symbols[self.board[row, 0]], "|", symbols[self.board[row, 1]], "|", symbols[self.board[row, 2]], "|")
            print("-------------")
class MinimaxAgent:
    def __init__(self, player):
        self.player = player
        self.cache = {}

    def board_key(self, board, current_player):
        return (tuple(board.flatten()), int(current_player))

    def get_best_move(self, board, current_player):
        best_score = -float('inf') if self.player == 1 else float('inf')
        best_move = None
        
        available_moves = [(r, c) for r in range(3) for c in range(3) if board[r, c] == 0]
        if not available_moves:
            return None

        alpha = -float('inf')
        beta = float('inf')

        for r, c in available_moves:
            temp_board = np.copy(board)
            temp_board[r, c] = current_player
            score = self.minimax(temp_board, 0, current_player * -1, alpha, beta)

            if self.player == 1:
                if score > best_score:
                    best_score = score
                    best_move = (r, c)
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = (r, c)
                beta = min(beta, best_score)
            
            if beta <= alpha:
                break
                
        return best_move

    def minimax(self, board, depth, current_player_in_search, alpha, beta):
        key = self.board_key(board, current_player_in_search)
        if key in self.cache:
            return self.cache[key]

        winner = self._check_win_status(board)
        if winner == 1: score = 10 - depth
        elif winner == -1: score = -10 + depth
        elif winner == 2: score = 0
        else: score = None

        if score is not None:
            self.cache[key] = score
            return score

        available_moves = [(r, c) for r in range(3) for c in range(3) if board[r, c] == 0]
        if not available_moves:
            score = 0
            self.cache[key] = score
            return score

        if current_player_in_search == 1:
            max_eval = -float('inf')
            for r, c in available_moves:
                temp_board = np.copy(board)
                temp_board[r, c] = 1
                eval_score = self.minimax(temp_board, depth + 1, -1, alpha, beta)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha: break 
            self.cache[key] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for r, c in available_moves:
                temp_board = np.copy(board)
                temp_board[r, c] = -1
                eval_score = self.minimax(temp_board, depth + 1, 1, alpha, beta)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha: break
            self.cache[key] = min_eval
            return min_eval

    def _check_win_status(self, board):
        for player_val in [1, -1]:
            for row in range(3):
                if np.all(board[row, :] == player_val): return player_val
            for col in range(3):
                if np.all(board[:, col] == player_val): return player_val
            if np.all(np.diag(board) == player_val) or np.all(np.diag(np.fliplr(board)) == player_val): return player_val
        if np.all(board != 0): return 2
        return 0

class CustomMinimaxAgent(MinimaxAgent):
    def __init__(self, player, random_move_prob):
        super().__init__(player)
        self.random_move_prob = random_move_prob

    def get_best_move(self, board, current_player):
        available_moves = [(r, c) for r in range(3) for c in range(3) if board[r, c] == 0]
        if not available_moves:
            return None
        
        if random.random() < self.random_move_prob:
            return random.choice(available_moves)
        else:
            return super().get_best_move(board, current_player)
def play_against_ai():
    while True:
        random_chance = -1
        while not (0 <= random_chance <= 100):
            try:
                user_input = input("Enter AI random move chance (0-100%): ")
                random_chance = int(user_input)
                if not (0 <= random_chance <= 100):
                    print("Invalid input. Please enter a number between 0 and 100.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        random_move_prob = random_chance / 100.0

        env = TicTacToe()
        
        ai_player_char = 'O'
        ai_player_val = -1
        human_player_char = 'X'
        human_player_val = 1
      
        ai_agent = CustomMinimaxAgent(player=ai_player_val, random_move_prob=random_move_prob)

        state = env.reset()
        done = False
        print(f"\n--- New Game: You ({human_player_char}) vs AI ({ai_player_char}) ---")
        env.render()
        
        if random.choice([True, False]):
            print(f"You ({human_player_char}) go first.")
            current_turn_player_val = human_player_val
        else:
            print(f"AI ({ai_player_char}) goes first.")
            current_turn_player_val = ai_player_val
        
        env.current_player = current_turn_player_val

        while not done:
            if env.current_player == human_player_val:
                available_moves = env.get_available_moves()
                valid_move = False
                while not valid_move:
                    try:
                        move_input = input(f"Your turn ({human_player_char}). Enter your move (row,col, e.g., 0,0): ")
                        r, c = map(int, move_input.split(','))
                        action = (r, c)
                        if action in available_moves:
                            valid_move = True
                        else:
                            print("Invalid move. Square is occupied or out of bounds. Try again.")
                    except (ValueError, IndexError):
                        print("Invalid format. Please enter as 'row,col'.")
                
                state, _, done, _ = env.make_move(action)
            else:
                print(f"AI ({ai_player_char}) is thinking...")
                board_np = np.array(state).reshape(3, 3)
                action = ai_agent.get_best_move(board_np, env.current_player)
                
                if action is None:
                    break
                print(f"AI ({ai_player_char}) plays: {action}")
                state, _, done, _ = env.make_move(action)
            
            env.render()
            
            if done:
                if env.winner == human_player_val:
                    print("You win!")
                elif env.winner == ai_player_val:
                    print("AI wins!")
                else:
                    print("It's a Draw!")
                break
        
        play_again = input("Play again? (y/n): ")
        if play_again.lower() != 'y':
            break
if __name__ == "__main__":
    play_against_ai()
