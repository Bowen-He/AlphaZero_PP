#!/usr/bin/env python3
"""Pretrain reactive network on oracle prefix dataset.

Trains a ReactivePolicyValueNet on oracle-derived (obs, policy, value) triples.
Reports policy accuracy and value RMSE.

Usage:
    python scripts/pretrain_reactive_network.py --dataset results/reactive_prefix_dataset/ --epochs 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alphazeropp.instances.doors.dsl.reactive_prefix_dataset import (
    ReactivePrefixDataset,
)
from alphazeropp.instances.doors.dsl.reactive_network import (
    ReactivePolicyValueNet,
)


def evaluate_network(net, examples):
    """Compute policy top-k accuracy and value RMSE."""
    top1_correct = 0
    top3_correct = 0
    value_errors = []

    for obs, (pi_target, v_target) in examples:
        policy, value = net.predict(obs)
        pred_top1 = np.argmax(policy)
        pred_top3 = set(np.argsort(policy)[-3:])
        target_best = set(np.where(pi_target > 0)[0])

        if pred_top1 in target_best:
            top1_correct += 1
        if pred_top3 & target_best:
            top3_correct += 1
        value_errors.append((value - v_target) ** 2)

    n = len(examples)
    return {
        "top1_acc": top1_correct / n if n > 0 else 0,
        "top3_acc": top3_correct / n if n > 0 else 0,
        "value_rmse": np.sqrt(np.mean(value_errors)) if value_errors else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain reactive network on oracle prefix data",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Directory containing D*_*.jsonl files",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/reactive_stage4_pretrained.pt",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-branches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load datasets
    dataset_dir = Path(args.dataset)
    datasets = []
    for path in sorted(dataset_dir.glob("*.jsonl")):
        print(f"Loading {path}...")
        datasets.append(ReactivePrefixDataset.load(path))

    if not datasets:
        print(f"No .jsonl files found in {dataset_dir}")
        sys.exit(1)

    merged = ReactivePrefixDataset.merge(*datasets)
    examples = merged.to_training_examples()
    print(f"Total training examples: {len(examples)}")

    # Create network
    n_steps = 2 * args.n_branches
    net = ReactivePolicyValueNet(
        budget=n_steps,
        action_size=7,
        d_model=64,
        n_heads=4,
        n_layers=2,
        aux_p_solve=True,
        training_params={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "policy_weight": 2.0,
        },
        random_seed=args.seed,
    )

    # Pre-training metrics
    metrics_before = evaluate_network(net, examples[:200])
    print(f"Before training: {metrics_before}")

    # Train
    t0 = time.time()

    # Build auxiliary examples for p_solve
    aux_examples = [
        {"p_solve": entry.p_solve}
        for entry in merged.entries
    ]

    net.train(examples, aux_examples=aux_examples)
    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.1f}s")

    # Post-training metrics
    metrics_after = evaluate_network(net, examples[:200])
    print(f"After training: {metrics_after}")

    # Save checkpoint
    checkpoint_dir = Path(args.checkpoint).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    net.save_checkpoint(str(checkpoint_dir))
    print(f"Saved checkpoint: {checkpoint_dir}")

    # Summary
    print(f"\n=== Pretraining Summary ===")
    print(f"  Examples:   {len(examples)}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Time:       {elapsed:.1f}s")
    print(f"  Top-1 acc:  {metrics_before['top1_acc']:.3f} → {metrics_after['top1_acc']:.3f}")
    print(f"  Top-3 acc:  {metrics_before['top3_acc']:.3f} → {metrics_after['top3_acc']:.3f}")
    print(f"  Value RMSE: {metrics_before['value_rmse']:.4f} → {metrics_after['value_rmse']:.4f}")


if __name__ == "__main__":
    main()
