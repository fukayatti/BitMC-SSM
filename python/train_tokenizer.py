"""
Multilingual (Japanese + English) Byte-Level BPE Tokenizer Trainer
====================================================================
Trains a single BPE vocabulary shared across Japanese and English so the
BitMC-SSM base model can be pretrained on a bilingual corpus instead of the
English-only GPT-2 vocabulary used by the original nano/micro/small tiers.

Uses the same byte-level pre-tokenization scheme as GPT-2 (raw UTF-8 bytes
mapped to a printable unicode alphabet before BPE), so the resulting
vocab.json / merges.txt are drop-in compatible with the existing C++
inference engine (src/infer.cpp) without any C++ changes.

Samples from three registers so the vocabulary compresses both formal and
colloquial Japanese well (plain Wikipedia BPE tends to fragment casual/slang
text into near byte-level tokens, which hurts the keyboard-input use case):
  - Japanese Wikipedia    (formal)
  - Japanese Web (CC-100) (colloquial blog-style)
  - English (SmolLM cosmopedia-v2)

Usage:
  python python/train_tokenizer.py \
      --vocab_size 49152 \
      --ja_wiki_docs 100000 \
      --ja_web_docs 100000 \
      --en_docs 200000 \
      --out_dir tokenizer
"""

import argparse
import os
import sys

from tokenizers import ByteLevelBPETokenizer


def _stream_docs(dataset_name, subset, n_docs, label):
    from datasets import load_dataset

    print(f"📚 Streaming {label} ({n_docs:,} docs)...")
    stream = load_dataset(dataset_name, subset, split="train", streaming=True) if subset else load_dataset(dataset_name, split="train", streaming=True)
    count = 0
    for item in stream:
        text = item.get("text", "").strip()
        if len(text) < 20:
            continue
        yield text
        count += 1
        if count >= n_docs:
            break
    print(f"✅ Collected {count:,} {label} documents.")


def stream_corpus(ja_wiki_docs: int, ja_web_docs: int, en_docs: int):
    yield from _stream_docs("wikimedia/wikipedia", "20231101.ja", ja_wiki_docs, "Japanese Wikipedia")
    yield from _stream_docs("range3/cc100-ja", None, ja_web_docs, "Japanese Web (CC-100)")
    yield from _stream_docs("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", en_docs, "English (SmolLM)")


def get_args():
    parser = argparse.ArgumentParser(description="Train a JA+EN Byte-Level BPE tokenizer for BitMC-SSM")
    parser.add_argument("--vocab_size", type=int, default=49152, help="Target vocabulary size (kept modest to stay edge-friendly)")
    parser.add_argument("--ja_wiki_docs", type=int, default=100000, help="Number of Japanese Wikipedia documents to sample")
    parser.add_argument("--ja_web_docs", type=int, default=100000, help="Number of Japanese CC-100 (colloquial web) documents to sample")
    parser.add_argument("--en_docs", type=int, default=200000, help="Number of English (SmolLM cosmopedia-v2) documents to sample")
    parser.add_argument("--out_dir", type=str, default="tokenizer", help="Output directory for vocab.json / merges.txt")
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        stream_corpus(args.ja_wiki_docs, args.ja_web_docs, args.en_docs),
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>"],
    )

    tokenizer.save_model(args.out_dir)

    eot_id = tokenizer.token_to_id("<|endoftext|>")

    import json
    with open(os.path.join(args.out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab_size": tokenizer.get_vocab_size(), "eot_token_id": eot_id}, f, indent=2)

    print("----------------------------------------------------------------------")
    print(f"✅ Tokenizer saved to {args.out_dir}/vocab.json + {args.out_dir}/merges.txt")
    print(f"📊 Vocab size : {tokenizer.get_vocab_size():,}")
    print(f"🔚 EOT token ID: {eot_id}  (auto-read by preprocess_data.py --tokenizer_dir, saved in tokenizer_config.json)")
    print("----------------------------------------------------------------------")
    sys.stdout.flush()
    # os._exit(0) avoids a PyGILState crash from datasets/pyarrow's background
    # streaming threads during normal interpreter shutdown (same workaround as
    # preprocess_data.py).
    os._exit(0)


if __name__ == "__main__":
    main()
