"""Reactive policy-value network with optional auxiliary heads.

Extends the DerivationTransformerModel with:
  - p_solve head: predicts probability of a solving completion
  - best_action head: predicts oracle best-next-action mask

The predict() method returns standard (policy, value) for MCTS compatibility.
Auxiliary heads are used only during training when oracle labels are available.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazeropp.core.policy_value_net import TorchPolicyValueNet
from alphazeropp.utils import get_device

logger = logging.getLogger(__name__)


class ReactiveTransformerModel(nn.Module):
    """Transformer encoder for reactive derivation game observations.

    Input:  flat obs tensor of shape (batch, 2 * budget).
    Output: dict with policy_logits, value, and optional auxiliary heads.
    """

    def __init__(
        self,
        budget: int,
        action_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        aux_p_solve: bool = True,
        aux_best_action: bool = False,
    ):
        super().__init__()
        self.budget = budget
        self.action_size = action_size
        seq_len = budget

        # Embeddings
        self.type_emb = nn.Embedding(num_embeddings=9, embedding_dim=d_model)
        self.param_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Embedding(num_embeddings=seq_len + 1, embedding_dim=d_model)
        self.cls_emb = nn.Parameter(torch.zeros(d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)

        # Core heads
        self.policy_head = nn.Linear(d_model, action_size)
        self.value_head = nn.Linear(d_model, 1)

        # Auxiliary heads
        self.p_solve_head = nn.Linear(d_model, 1) if aux_p_solve else None
        self.best_action_head = nn.Linear(d_model, action_size) if aux_best_action else None

    def forward(self, x):
        B = x.shape[0]
        L = self.budget

        type_ids = x[:, 0::2].long()
        params = x[:, 1::2]

        pad_mask_tokens = (type_ids == 0)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([cls_mask, pad_mask_tokens], dim=1)

        type_e = self.type_emb(type_ids)
        param_e = self.param_proj(params.unsqueeze(-1))
        token_emb = type_e + param_e

        cls = self.cls_emb.unsqueeze(0).expand(B, -1).unsqueeze(1)
        seq = torch.cat([cls, token_emb], dim=1)
        positions = torch.arange(L + 1, device=x.device)
        seq = seq + self.pos_emb(positions)

        hidden = self.transformer(seq, src_key_padding_mask=key_padding_mask)
        cls_out = self.final_norm(hidden[:, 0, :])

        result = {
            "policy_logits": self.policy_head(cls_out),
            "value": self.value_head(cls_out).squeeze(-1),
        }
        if self.p_solve_head is not None:
            result["p_solve"] = torch.sigmoid(self.p_solve_head(cls_out).squeeze(-1))
        if self.best_action_head is not None:
            result["best_action"] = torch.sigmoid(self.best_action_head(cls_out))

        return result


class ReactivePolicyValueNet(TorchPolicyValueNet):
    """Reactive policy-value network with auxiliary head support.

    Follows the same interface as DerivationPolicyValueNet for MCTS
    compatibility. Auxiliary heads are used only during training.
    """

    save_file_name = "reactive_checkpoint.pt"

    default_training_params = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "policy_weight": 2.0,
        "p_solve_weight": 0.5,
        "best_action_weight": 0.5,
    }

    def __init__(
        self,
        budget: int,
        action_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        aux_p_solve: bool = True,
        aux_best_action: bool = False,
        training_params: dict = {},
        random_seed: int | None = None,
        device=None,
    ):
        if random_seed is not None:
            torch.manual_seed(random_seed)

        model = ReactiveTransformerModel(
            budget=budget,
            action_size=action_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            aux_p_solve=aux_p_solve,
            aux_best_action=aux_best_action,
        )
        self.budget = budget
        self.action_size = action_size
        super().__init__(model)
        self.training_params = self.default_training_params | training_params
        self.DEVICE = get_device() if device is None else device

    def predict(self, state):
        """Standard MCTS interface: returns (policy_probs, value).

        Returns value as a 0-d numpy array (not Python float) to match
        the DerivationPolicyValueNet contract expected by MCTS.
        """
        if next(self.model.parameters()).device.type != "cpu":
            self.model.cpu()
        nn_input = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            out = self.model(nn_input)
            policy_prob = F.softmax(out["policy_logits"], dim=-1)

        policy_prob = policy_prob.numpy().squeeze(0)
        value = out["value"].numpy().squeeze(0)

        assert policy_prob.shape == (self.action_size,)
        assert value.shape == ()
        return policy_prob, value

    def train(self, examples, needs_reshape=True, print_all_epochs=False):
        """Train on examples in the standard AlphaZero format.

        Args:
            examples: List of (state, policy, value) tuples — same format as
                      DerivationPolicyValueNet.train().
            needs_reshape: If True, examples are (state, policy, value) tuples.
            print_all_epochs: Unused, kept for interface compatibility.

        Returns:
            (model, batch_losses, train_losses, policy_losses, value_losses)
            to match DerivationPolicyValueNet contract.
        """
        if len(examples) == 0:
            return self.model, [], [], [], []

        tp = self.training_params
        self.model.to(self.DEVICE)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=tp["learning_rate"],
            weight_decay=tp["weight_decay"],
        )

        # Parse examples: (state, policy, value) format
        if needs_reshape:
            states = torch.tensor(
                np.array([s for s, _, _ in examples], dtype=np.float32),
            ).to(self.DEVICE)
            target_pis = torch.tensor(
                np.array([p for _, p, _ in examples], dtype=np.float32),
            ).to(self.DEVICE)
            target_vs = torch.tensor(
                np.array([float(v) for _, _, v in examples], dtype=np.float32),
            ).to(self.DEVICE)
        else:
            raise NotImplementedError("needs_reshape=False not supported")

        n = len(examples)
        bs = tp["batch_size"]
        policy_weight = tp["policy_weight"]

        train_batch_losses = []
        train_losses = []
        policy_losses = []
        value_losses = []

        for epoch in range(tp["epochs"]):
            self.model.train()
            perm = torch.randperm(n, device=self.DEVICE)
            epoch_loss = 0.0
            epoch_policy = 0.0
            epoch_value = 0.0
            n_batches = 0

            for i in range(0, n, bs):
                idx = perm[i: i + bs]
                s = states[idx]
                pi = target_pis[idx]
                v = target_vs[idx]

                out = self.model(s)

                log_probs = F.log_softmax(out["policy_logits"], dim=-1)
                p_loss = -torch.sum(pi * log_probs) / pi.shape[0]
                v_loss = F.mse_loss(out["value"], v)
                loss = policy_weight * p_loss + v_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_batch_losses.append(loss.item())
                epoch_loss += loss.item()
                epoch_policy += p_loss.item()
                epoch_value += v_loss.item()
                n_batches += 1

            if n_batches > 0:
                train_losses.append(epoch_loss / n_batches)
                policy_losses.append(epoch_policy / n_batches)
                value_losses.append(epoch_value / n_batches)

        self.model.cpu()
        return self.model, train_batch_losses, train_losses, policy_losses, value_losses

    def train_with_aux(self, examples, aux_p_solve_values=None):
        """Train with optional auxiliary p_solve targets.

        Args:
            examples: List of (state, policy, value) tuples.
            aux_p_solve_values: Optional list of float p_solve targets (same length).
        """
        # First do standard training
        result = self.train(examples)

        # Then auxiliary pass if provided
        if aux_p_solve_values is not None and self.model.p_solve_head is not None:
            tp = self.training_params
            self.model.to(self.DEVICE)
            self.model.train()
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=tp["learning_rate"] * 0.5,
                weight_decay=tp["weight_decay"],
            )

            states = torch.tensor(
                np.array([s for s, _, _ in examples], dtype=np.float32),
            ).to(self.DEVICE)
            targets = torch.tensor(
                np.array(aux_p_solve_values, dtype=np.float32),
            ).to(self.DEVICE)

            bs = tp["batch_size"]
            for epoch in range(max(1, tp["epochs"] // 2)):
                perm = torch.randperm(len(examples), device=self.DEVICE)
                for i in range(0, len(examples), bs):
                    idx = perm[i: i + bs]
                    out = self.model(states[idx])
                    loss = F.binary_cross_entropy(out["p_solve"], targets[idx])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            self.model.cpu()

        return result
