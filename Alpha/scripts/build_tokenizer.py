#!/usr/bin/env python3
"""Train Alpha SPM Unigram tokenizer (local mirror; notebook Cell 02 is standalone).

Usage: python3 scripts/build_tokenizer.py --out alpha_tokenizer --vocab-size 48000
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="text file, one doc per line")
    ap.add_argument("--out", default="alpha_tokenizer")
    ap.add_argument("--vocab-size", type=int, default=48000)
    args = ap.parse_args()
    import sentencepiece as spm
    spm.SentencePieceTrainer.train(
        input=args.corpus, model_prefix=args.out, vocab_size=args.vocab_size,
        model_type="unigram", character_coverage=1.0,
        input_sentence_size=5_000_000, shuffle_input_sentence=True,
        hard_vocab_limit=False, unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
    )
    print(f"wrote {args.out}.model", flush=True)


if __name__ == "__main__":
    main()
