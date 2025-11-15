import torch
import torch.nn.functional as F


def supervised_step(model, optimizer, batch_positions, batch_targets, batch_outcomes=None):
    """
    This is the training step function.
    It's designed to train *only* the Policy Head if outcomes are not provided.
    """
    model.train()
    logits, values = model(batch_positions)  # Get both heads

    # --- Policy Loss ---
    loss_policy = F.cross_entropy(logits, batch_targets)

    # --- Value Loss (THE HACK) ---
    # We will pass batch_outcomes=None.
    # This ensures loss_value is 0.0 and we *only* train the Policy Head.
    loss_value = torch.tensor(0.0).to(batch_positions.device)
    if batch_outcomes is not None:
        loss_value = F.mse_loss(values, batch_outcomes)

    loss = loss_policy + loss_value

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), loss_policy.item(), loss_value.item()