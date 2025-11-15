import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..nueral_network import Anon
from .pgn_dataset import PGNDataset

def supervised_step(model, optimizer, batch_positions, batch_targets, batch_outcomes=None):
    model.train() # Set model(NN) to training mode
    logits, values = model(batch_positions) # perform forward pass for two-headed network
    loss_policy = F.cross_entropy(logits, batch_targets) # Classification, cross_entropy is standard function for this
    loss_value = torch.tensor(0.0)
    if batch_outcomes is not None:
        loss_value = F.mse_loss(values, batch_outcomes) # Mean Squared Error loss function in regression evaluation
    loss = loss_policy + loss_value
    # Find a single set of weights that minimize the combined error of both the
    # policy and value heads
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), loss_policy.item(), loss_value.item() if isinstance(loss_value, torch.Tensor) else float(loss_value)

def collate(batch):
    positions = torch.stack([item[0] for item in batch], dim=0)
    # Takes all the individual tensors from the batch and "stacks" them into a single (batch_size, 18, 8, 8)
    # tensor, which is what the model expects.
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    # Takes a 1D tensor of target moves indices.
    outcomes = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    # Creates a 1D tensor of game outcomes
    return positions, targets, outcomes


def tiny_supervised_train(pgn_path: str = None):
    if pgn_path is None:
        print("No PGN provided. Use --pgn <file> to train on a small PGN subset.")
        return
    ds = PGNDataset(pgn_path, max_games=20, max_moves_per_game=100)
    if len(ds) == 0:
        print("Dataset empty. Check PGN or use --demo")
        return
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=lambda batch: collate(batch))
    model = Anon()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(3):
        for i, (pos_batch, tgt_batch, out_batch) in enumerate(loader):
            loss, p_loss, v_loss = supervised_step(model, opt, pos_batch, tgt_batch, out_batch)
            if i % 10 == 0:
                print(f"epoch {epoch} iter {i} loss {loss:.4f} p:{p_loss:.4f} v:{v_loss:.4f}")
    print("Done tiny train (toy)")
