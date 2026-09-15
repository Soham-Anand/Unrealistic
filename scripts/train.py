#!/usr/bin/env python3
"""Train Unrealistic LLM from scratch."""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.trainer import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", default="configs/training_config.json",
                        help="Path to training config JSON")
    parser.add_argument("--model-config", default="configs/model_config.json",
                        help="Path to model config JSON")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start from scratch instead of resuming latest checkpoint")
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="Discard saved optimizer state and restart LR warmup (use at phase starts)")
    args = parser.parse_args()

    train(config_path=args.model_config,
          train_config_path=args.train_config,
          resume=not args.no_resume,
          reset_optimizer=args.reset_optimizer)


if __name__ == "__main__":
    main()
