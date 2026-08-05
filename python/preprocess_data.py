"""
Pre-tokenization Pipeline for BitMC-SSM
=======================================
Converts raw streaming datasets (TinyStories, SmolLM, custom txt/jsonl)
into high-performance memory-mapped binary (.bin) files (uint16 array).

Usage:
  python python/preprocess_data.py --dataset tinystories --num_samples 50000 --out data/tinystories_50k.bin
"""

import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm


def get_tokenizer():
    """Returns a fast GPT-2 BPE tokenizer using tiktoken or transformers."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        print("⚡ Using Tiktoken (gpt2 / r50k_base) for ultra-fast preprocessing.")
        return lambda text: enc.encode(text, allowed_special={"<|endoftext|>"})
    except ImportError:
        try:
            from transformers import GPT2TokenizerFast
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            print("📦 Using transformers GPT2TokenizerFast.")
            return lambda text: tokenizer.encode(text)
        except ImportError:
            print("❌ Error: Please install tiktoken or transformers (e.g. 'pip install tiktoken').")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="BitMC-SSM Binary Pre-tokenization Tool")
    parser.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories", "smollm", "custom"], help="Dataset source")
    parser.add_argument("--dataset_subset", type=str, default="stories", help="SmolLM subset (stories, cosmopedia-v2, etc.)")
    parser.add_argument("--input_file", type=str, default=None, help="Path to custom .txt or .jsonl file")
    parser.add_argument("--out", type=str, default="data/train_tokens.bin", help="Output .bin file path")
    parser.add_argument("--num_samples", type=int, default=50000, help="Maximum number of samples/documents to process (-1 for all)")
    parser.add_argument("--eot_token", type=int, default=50256, help="End-of-text token ID (50256 for GPT-2)")
    parser.add_argument("--flush_every", type=int, default=50000, help="Flush to disk every N tokens")
    args = parser.parse_args()

    encode_fn = get_tokenizer()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("======================================================================")
    print("⚡ Bit-MC-SSM High-Performance Pre-tokenization Engine")
    print("======================================================================")
    print(f"📁 Target Output : {args.out}")
    print(f"📚 Dataset Source: {args.dataset} (subset={args.dataset_subset if args.dataset == 'smollm' else 'default'})")
    print(f"🔢 Target Samples: {'Unlimited (all)' if args.num_samples <= 0 else args.num_samples}")
    print("----------------------------------------------------------------------")

    # Load dataset stream
    if args.dataset in ["tinystories", "smollm"]:
        try:
            from datasets import load_dataset
        except ImportError:
            print("❌ Error: 'datasets' package is required. Install via 'pip install datasets'.")
            sys.exit(1)

        if args.dataset == "tinystories":
            raw_stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        else:
            raw_stream = load_dataset("HuggingFaceTB/smollm-corpus", args.dataset_subset, split="train", streaming=True)
    elif args.dataset == "custom":
        if not args.input_file or not os.path.exists(args.input_file):
            print(f"❌ Error: Custom input file not found: {args.input_file}")
            sys.exit(1)

        def custom_stream_generator():
            import json
            with open(args.input_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if args.input_file.endswith(".jsonl"):
                        try:
                            obj = json.loads(line)
                            yield {"text": obj.get("text", "")}
                        except Exception:
                            continue
                    else:
                        yield {"text": line}

        raw_stream = custom_stream_generator()

    t0 = time.time()
    total_tokens = 0
    total_docs = 0
    token_buffer = []

    # Open binary file for writing
    with open(args.out, "wb") as f_out:
        pbar = tqdm(desc="Tokenizing & Packing (uint16)", unit=" docs")
        for item in raw_stream:
            text = item.get("text", "").strip()
            if len(text) < 10:
                continue

            toks = encode_fn(text)
            if not toks:
                continue

            # Append tokens and EOT separator
            token_buffer.extend(toks)
            if args.eot_token >= 0:
                token_buffer.append(args.eot_token)

            total_docs += 1
            pbar.update(1)

            # Flush buffer in chunks to keep memory usage tiny
            if len(token_buffer) >= args.flush_every:
                arr = np.array(token_buffer, dtype=np.uint16)
                f_out.write(arr.tobytes())
                total_tokens += len(token_buffer)
                token_buffer = []
                pbar.set_postfix({"tokens": f"{total_tokens:,}"})

            if args.num_samples > 0 and total_docs >= args.num_samples:
                break

        # Flush remaining buffer
        if token_buffer:
            arr = np.array(token_buffer, dtype=np.uint16)
            f_out.write(arr.tobytes())
            total_tokens += len(token_buffer)
            token_buffer = []

        pbar.close()

    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(args.out) / (1024 * 1024)
    tokens_per_sec = total_tokens / max(elapsed, 1e-4)

    print("\n----------------------------------------------------------------------")
    print(f"✅ Pre-tokenization complete in {elapsed:.2f} seconds!")
    print(f"📦 Output File     : {args.out}")
    print(f"📊 Total Documents : {total_docs:,}")
    print(f"⚡ Total Tokens    : {total_tokens:,}")
    print(f"💾 Binary Size     : {file_size_mb:.2f} MB")
    print(f"🚀 Speed           : {tokens_per_sec:,.0f} tokens / sec")
    print("======================================================================\n")
    sys.stdout.flush()
    sys.stderr.flush()
    # Use os._exit(0) to prevent pyarrow background streaming threads from crashing on Python GIL shutdown
    os._exit(0)


if __name__ == "__main__":
    main()
