"""
main_comat.py — Evaluation script for CoMAT-trained models (Q5).

Mirrors evaluation/main.py exactly, but uses gsm8k_comat.py which calls
the CoMAT system instruction at inference time.

Usage:

  # Evaluate CoMAT-SFT only
  python evaluation/main_comat.py \
      --model_signature Qwen/Qwen2.5-0.5B-Instruct \
      --sft_adapter_path ./checkpoints/Qwen/Qwen2.5-0.5B-Instruct-comat-sft \
      --output_path ./outputs/comat-sft

  # Evaluate CoMAT-SFT + GRPO
  python evaluation/main_comat.py \
      --model_signature Qwen/Qwen2.5-0.5B-Instruct \
      --sft_adapter_path ./checkpoints/Qwen/Qwen2.5-0.5B-Instruct-comat-sft \
      --grpo_adapter_path ./checkpoints/Qwen/Qwen2.5-0.5B-Instruct-comat-sft-grpo \
      --output_path ./outputs/comat-sft-grpo

  # Zero-shot with CoMAT instruction (ablation)
  python evaluation/main_comat.py \
      --model_signature Qwen/Qwen2.5-0.5B-Instruct \
      --output_path ./outputs/comat-zero-shot
"""

import json
import os
import re
import argparse
import time
import random
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from gsm8k_comat import load_gsm8k_questions, process_gsm8k_questions_comat


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def main():
    set_seed(42)

    parser = argparse.ArgumentParser(
        description="Evaluate CoMAT-trained models on GSM8K (Q5)"
    )
    parser.add_argument("--model_signature", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Base HuggingFace model.")
    parser.add_argument("--sft_adapter_path", required=False,
                        help="Path to the CoMAT SFT adapter.")
    parser.add_argument("--grpo_adapter_path", required=False,
                        help="Path to the GRPO adapter trained on top of CoMAT-SFT.")
    parser.add_argument("--output_path", required=True,
                        help="Directory to save evaluation results.")
    args = parser.parse_args()

    # Determine training mode
    if args.sft_adapter_path and args.grpo_adapter_path is None:
        training = "comat-sft"
    elif args.sft_adapter_path and args.grpo_adapter_path:
        training = "comat-sft+grpo"
    elif args.sft_adapter_path is None and args.grpo_adapter_path is None:
        training = "comat-zero-shot"
    else:
        raise ValueError(
            "Invalid combination. Provide --sft_adapter_path only (SFT), "
            "both adapters (SFT+GRPO), or neither (zero-shot)."
        )

    OUTPUT_DIR = args.output_path
    output_file_path = f"{OUTPUT_DIR}/results.json"
    log_file_path = f"{OUTPUT_DIR}/results.txt"

    ensure_dir(output_file_path)

    with open(output_file_path, 'w') as f:
        json.dump([], f)
    with open(log_file_path, 'w') as f:
        f.write(f"Start evaluating the {training} experiment.\n")
        f.write(f"Args: {args}\n")

    start_time = time.time()

    # ------------------------------------------------------------------ #
    #  Load model                                                          #
    # ------------------------------------------------------------------ #
    if training == "comat-sft+grpo":
        with open(log_file_path, 'a') as f:
            f.write(
                f"Evaluating {args.model_signature} with CoMAT-SFT from "
                f"{args.sft_adapter_path} and GRPO from {args.grpo_adapter_path}.\n"
            )
        base_id = args.model_signature
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        print("Merging CoMAT-SFT adapter...")
        model = PeftModel.from_pretrained(base_model, args.sft_adapter_path)
        model = model.merge_and_unload()
        print("Loading GRPO adapter...")
        model = PeftModel.from_pretrained(model, args.grpo_adapter_path)
        model_id = f"{base_id}+CoMAT-SFT+GRPO"
        model.eval()

    elif training == "comat-sft":
        base_id = args.model_signature
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        print("Loading CoMAT-SFT adapter...")
        model = PeftModel.from_pretrained(base_model, args.sft_adapter_path)
        model_id = f"{base_id}+CoMAT-SFT"
        model.eval()

    else:  # comat-zero-shot
        model_id = args.model_signature
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ------------------------------------------------------------------ #
    #  Evaluate                                                            #
    # ------------------------------------------------------------------ #
    dataset = load_from_disk("dataset/gsm8k_test_100")
    questions = load_gsm8k_questions(dataset)

    results, overall_accuracy, valid_accuracy, invalid_rate = process_gsm8k_questions_comat(
        questions,
        output_file_path,
        None,
        model_id,
        model,
        tokenizer,
        None,
    )

    end_time = time.time()
    duration = end_time - start_time

    print(f"\nFinal results saved to {output_file_path}")
    print(f"Overall Accuracy (including invalid): {overall_accuracy:.2%}")
    print(f"Valid Accuracy   (excluding invalid): {valid_accuracy:.2%}")
    print(f"Invalid Rate:                         {invalid_rate:.2%}")
    print(f"Evaluation Duration:                  {duration:.2f} seconds")

    with open(log_file_path, 'a') as f:
        f.write(f"Overall Accuracy (including invalid): {overall_accuracy:.2%}\n")
        f.write(f"Valid Accuracy (excluding invalid): {valid_accuracy:.2%}\n")
        f.write(f"Invalid Rate: {invalid_rate:.2%}\n")
        f.write(f"Evaluation Duration: {duration:.2f} seconds\n")

    print(f"Log file updated: {log_file_path}")


if __name__ == "__main__":
    main()
