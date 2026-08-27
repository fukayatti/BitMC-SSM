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


def get_tokenizer(tokenizer_dir=None):
    """Returns a fast BPE tokenizer. Uses a custom-trained multilingual
    ByteLevelBPETokenizer (see train_tokenizer.py) if `tokenizer_dir` is given,
    otherwise falls back to the English-only GPT-2 vocabulary."""
    if tokenizer_dir:
        vocab_file = os.path.join(tokenizer_dir, "vocab.json")
        merges_file = os.path.join(tokenizer_dir, "merges.txt")
        if not (os.path.exists(vocab_file) and os.path.exists(merges_file)):
            print(f"❌ Error: vocab.json/merges.txt not found in '{tokenizer_dir}'. Run python/train_tokenizer.py first.")
            sys.exit(1)
        from tokenizers import ByteLevelBPETokenizer
        tokenizer = ByteLevelBPETokenizer.from_file(vocab_file, merges_file)
        print(f"🌏 Using custom multilingual tokenizer from '{tokenizer_dir}' (vocab_size={tokenizer.get_vocab_size():,}).")
        return lambda text: tokenizer.encode(text).ids

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


def ja_en_mix_stream():
    """Weighted round-robin over three registers so the base model doesn't
    only learn stiff encyclopedic text:
      - Japanese Wikipedia   (formal, factual grounding)
      - Japanese Web (CC-100) (colloquial blog-style text -- crucial for the
        keyboard-input/IME use case, which models everyday casual Japanese)
      - English (SmolLM cosmopedia-v2, already more varied than raw Wikipedia)
    English gets ~50% weight, Japanese ~50% split evenly between the two
    registers above, and streams are interleaved (not concatenated) so a
    budget-limited run that stops early still saw all three throughout."""
    from datasets import load_dataset

    # HF streaming iterates each dataset in on-disk shard order, not randomly;
    # a local shuffle buffer avoids the base model only ever seeing whatever
    # narrow, possibly domain-clustered slice happens to be stored first.
    ja_wiki = iter(load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True).shuffle(seed=42, buffer_size=10000))
    ja_web = iter(load_dataset("range3/cc100-ja", split="train", streaming=True).shuffle(seed=42, buffer_size=10000))
    en_stream = iter(load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True).shuffle(seed=42, buffer_size=10000))

    # en_stream appears twice so English gets 2x the pull rate of each
    # individual Japanese source (i.e. EN 50% / ja_wiki 25% / ja_web 25%).
    schedule = [en_stream, ja_wiki, en_stream, ja_web]
    exhausted = set()

    idx = 0
    while len(exhausted) < len({id(s) for s in schedule}):
        stream = schedule[idx % len(schedule)]
        idx += 1
        if id(stream) in exhausted:
            continue
        try:
            yield next(stream)
        except StopIteration:
            exhausted.add(id(stream))


def main():
    parser = argparse.ArgumentParser(description="BitMC-SSM Binary Pre-tokenization Tool")
    parser.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories", "smollm", "ja_en_mix", "custom"], help="Dataset source")
    parser.add_argument("--dataset_subset", type=str, default="stories", help="SmolLM subset (stories, cosmopedia-v2, etc.)")
    parser.add_argument("--input_file", type=str, default=None, help="Path to custom .txt or .jsonl file")
    parser.add_argument("--out", type=str, default="data/train_tokens.bin", help="Output .bin file path")
    parser.add_argument("--num_samples", type=int, default=50000, help="Maximum number of samples/documents to process (-1 for all)")
    parser.add_argument("--eot_token", type=int, default=None, help="End-of-text token ID. Defaults to 50256 (GPT-2), or auto-read from --tokenizer_dir's tokenizer_config.json if present.")
    parser.add_argument("--flush_every", type=int, default=50000, help="Flush to disk every N tokens")
    parser.add_argument("--tokenizer_dir", type=str, default=None, help="Directory with a custom vocab.json/merges.txt from train_tokenizer.py (defaults to English-only GPT-2 vocab if omitted)")
    args = parser.parse_args()

    encode_fn = get_tokenizer(args.tokenizer_dir)

    if args.eot_token is None:
        args.eot_token = 50256
        if args.tokenizer_dir:
            config_path = os.path.join(args.tokenizer_dir, "tokenizer_config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                args.eot_token = config["eot_token_id"]
                print(f"🔚 Auto-detected EOT token ID from tokenizer_config.json: {args.eot_token}")

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
    elif args.dataset == "ja_en_mix":
        raw_stream = ja_en_mix_stream()
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
