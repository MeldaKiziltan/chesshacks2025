import torch
import torch.nn.functional as F
import chess
import math
import time
import numpy as np
import random

from neural_network import Anon
from features import board_to_tensor, move_to_index, index_to_move
from utils.GameContext import GameContext
from utils import chess_manager

# --- 1. Load The Model (This happens ONCE at build time) ---
print("--- Loading Model for MCTS Bot ---")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = Anon()  #  `Anon` model
# This is the trained model file from  Modal job
MODEL_PATH = "best_model.pt"
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
if 'model_state' in state_dict:
    state_dict = state_dict['model_state']
MODEL.load_state_dict(state_dict)
MODEL.to(DEVICE)
MODEL.eval()  # Set to evaluation mode
print("--- Model Loaded Successfully ---")


# --- 2. The MCTS Node and Searcher ---

class MCTSNode:
    """ This is one "node" in our game tree. """

    def __init__(self, parent=None, prior_p=0.0):
        self.parent = parent
        self.children = {}  # A map from chess.Move -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0  # This is the "win" score from random playouts
        self.prior_p = prior_p  # The "P" (Policy) from the Neural Network

    def get_value(self):
        if self.visit_count == 0: return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self, c_puct=1.5):
        """ Selects the child with the highest UCB score (PUCT algorithm). """
        best_score, best_move, best_child = -float('inf'), None, None

        # This is the "PUCT" formula, the core of AlphaZero's selection
        # It balances "exploitation" (get_value) and "exploration" (prior_p)
        for move, child in self.children.items():
            score = child.get_value() + c_puct * child.prior_p * \
                    (math.sqrt(self.parent.visit_count) / (1 + child.visit_count))
            if score > best_score:
                best_score, best_move, best_child = score, move, child
        return best_move, best_child

    def expand(self, board: chess.Board, policy_probs: torch.Tensor):
        """ Expand this node by creating all legal children. """
        for move in board.legal_moves:
            if move not in self.children:
                try:
                    # Get the *new* 4672 label for this legal move
                    label = move_to_index(move)
                    # Set the "prior" from our NN's gut instinct
                    self.children[move] = MCTSNode(parent=self, prior_p=policy_probs[label].item())
                except:
                    pass  # Ignore moves not in our 4672 map

    def backpropagate(self, value):
        """ Backpropagate the "value" up the tree. """
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            # We "flip" the value for the parent, since it's their opponent's move
            self.parent.backpropagate(-value)


def get_nn_policy(board: chess.Board) -> torch.Tensor:
    """ Gets the NN's "gut instinct" policy vector. """
    tensor = board_to_tensor(board).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        # Get both heads, but we only use the policy (logits)
        logits, _ = MODEL(tensor)
    return F.softmax(logits.squeeze(0), dim=0)  # (4672,)


def run_mcts_search(board: chess.Board, num_simulations=100):
    """
    This is the main MCTS loop that combines search and NN.
    """
    root_node = MCTSNode()

    # Get the NN's "gut instinct" for the root position
    # This sets the `prior_p` for all the first moves
    root_policy = get_nn_policy(board)
    root_node.expand(board, root_policy)

    for _ in range(num_simulations):
        node = root_node
        sim_board = board.copy()

        # 1. Selection
        while not node.is_leaf():
            move, node = node.select_child()
            if move is None: break
            sim_board.push(move)

        # 2. Expansion
        if not sim_board.is_game_over():
            policy = get_nn_policy(sim_board)
            node.expand(sim_board, policy)

        # 3. Simulation (Random Playout)
        # This is our "Value". We play random moves to see who wins.
        # This is a "from scratch" method, not using a Value Head.
        playout_board = sim_board.copy()
        for _ in range(15):  # Limit playout to 15 moves for speed
            if playout_board.is_game_over(): break
            try:
                playout_board.push(random.choice(list(playout_board.legal_moves)))
            except:
                break

        # Get the result from White's perspective
        value = 0.0
        if playout_board.is_game_over():
            result = playout_board.result()
            if result == "1-0":
                value = 1.0
            elif result == "0-1":
                value = -1.0

        # 4. Backpropagation
        # We must ensure the value is from the *current node's* perspective
        if sim_board.turn == chess.BLACK:
            value = -value

        node.backpropagate(value)

    # After all simulations, pick the *most visited* move
    if not root_node.children:
        return random.choice(list(board.legal_moves))

    best_move = max(root_node.children, key=lambda move: root_node.children[move].visit_count)
    return best_move


# --- 3. The Hackathon Entrypoint ---
@chess_manager.entrypoint
def select_move(ctx: GameContext) -> chess.Move:
    """
    This is the main function called by the hackathon.
    """
    # Simple time management: 1/30th of our remaining time
    simulations_to_run = 100  # Default
    if ctx.time and ctx.time.my_ms > 10000:  # 10 seconds
        think_time_ms = ctx.time.my_ms / 30
        # A rough guess: 1000 simulations per second? We must tune this.
        simulations_to_run = int(think_time_ms)
        if simulations_to_run < 50: simulations_to_run = 50

    move = run_mcts_search(ctx.board, simulations_to_run)

    if move is None:
        return chess.Move.null()

    return move