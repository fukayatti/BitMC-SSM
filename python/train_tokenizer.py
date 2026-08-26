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

Each source is streamed once and cached to a local .txt file as it downloads
(one line per document) so a) progress is visible (a line every 2,000 docs
instead of a silent multi-minute gap) and b) a disconnected/interrupted run
can resume instantly from the cache on the next invocation instead of
re-streaming from HuggingFace from scratch.

Usage:
  python python/train_tokenizer.py \
      --vocab_size 49152 \
      --ja_wiki_docs 20000 \
      --ja_web_docs 20000 \
      --en_docs 40000 \
      --out_dir tokenizer
"""

import argparse
import os
import sys

from tokenizers import ByteLevelBPETokenizer


def _stream_docs(dataset_name, subset, n_docs, label, cache_file):
    if os.path.exists(cache_file):
        print(f"📦 Using cached {label} corpus ({cache_file})")
        count = 0
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    yield line
                    count += 1
        print(f"✅ Loaded {count:,} cached {label} documents.")
        return

    from datasets import load_dataset

    print(f"📚 Streaming {label} ({n_docs:,} docs)...")
    stream = load_dataset(dataset_name, subset, split="train", streaming=True) if subset else load_dataset(dataset_name, split="train", streaming=True)
    count = 0
    tmp_file = cache_file + ".partial"
    with open(tmp_file, "w", encoding="utf-8") as out_f:
        for item in stream:
            text = item.get("text", "").strip()
            if len(text) < 20:
                continue
            out_f.write(text.replace("\n", " ") + "\n")
            yield text
            count += 1
            if count % 2000 == 0:
                print(f"   ...{count:,}/{n_docs:,} {label} docs")
                sys.stdout.flush()
            if count >= n_docs:
                break
    os.rename(tmp_file, cache_file)
    print(f"✅ Collected {count:,} {label} documents (cached to {cache_file}).")


def stream_corpus(ja_wiki_docs: int, ja_web_docs: int, en_docs: int, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    yield from _stream_docs("wikimedia/wikipedia", "20231101.ja", ja_wiki_docs, "Japanese Wikipedia", os.path.join(cache_dir, "ja_wiki.txt"))
    yield from _stream_docs("range3/cc100-ja", None, ja_web_docs, "Japanese Web (CC-100)", os.path.join(cache_dir, "ja_web.txt"))
    yield from _stream_docs("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", en_docs, "English (SmolLM)", os.path.join(cache_dir, "en.txt"))


def get_args():
    parser = argparse.ArgumentParser(description="Train a JA+EN Byte-Level BPE tokenizer for BitMC-SSM")
    parser.add_argument("--vocab_size", type=int, default=49152, help="Target vocabulary size (kept modest to stay edge-friendly)")
    parser.add_argument("--ja_wiki_docs", type=int, default=20000, help="Number of Japanese Wikipedia documents to sample")
    parser.add_argument("--ja_web_docs", type=int, default=20000, help="Number of Japanese CC-100 (colloquial web) documents to sample")
    parser.add_argument("--en_docs", type=int, default=40000, help="Number of English (SmolLM cosmopedia-v2) documents to sample")
    parser.add_argument("--out_dir", type=str, default="tokenizer", help="Output directory for vocab.json / merges.txt")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache streamed corpus text (defaults to <out_dir>/_corpus_cache). Delete cached files to force a fresh re-download with different doc counts.")
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(args.out_dir, "_corpus_cache")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        stream_corpus(args.ja_wiki_docs, args.ja_web_docs, args.en_docs, cache_dir),
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
