import re
from collections import Counter
from pathlib import Path


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer if present."""
    start_match = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE)
    end_match = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE)

    start_idx = start_match.end() if start_match else 0
    end_idx = end_match.start() if end_match else len(text)
    return text[start_idx:end_idx]


def load_and_tokenize(data_dir: str = "data") -> list[list[str]]:
    """Read all .txt files in data_dir and return a list of tokenized sentences."""
    sentences = []
    for path in sorted(Path(data_dir).glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        clean = strip_gutenberg_boilerplate(raw)

        # Split into rough sentences on ., !, or ?
        for chunk in re.split(r"[.!?]+", clean):
            tokens = re.findall(r"[a-z']+|[.,;]", chunk.lower())
            if tokens:
                sentences.append(tokens)
    return sentences


def split_and_map_unk(
    sentences: list[list[str]],
    train_ratio: float = 0.9,
    min_freq: int = 2,
) -> tuple[list[list[str]], list[list[str]], set[str]]:
    """Split into train/test BEFORE building vocab, then replace rare/unseen words with [UNK]."""
    split_idx = int(len(sentences) * train_ratio)
    train_raw = sentences[:split_idx]
    test_raw = sentences[split_idx:]

    # Count words on training data ONLY to prevent test leakage
    word_counts = Counter(word for sent in train_raw for word in sent)
    vocab = {word for word, count in word_counts.items() if count >= min_freq}
    vocab.add("[UNK]")

    train_sents = [[w if w in vocab else "[UNK]" for w in sent] for sent in train_raw]
    test_sents = [[w if w in vocab else "[UNK]" for w in sent] for sent in test_raw]

    return train_sents, test_sents, vocab


if __name__ == "__main__":
    all_sents = load_and_tokenize("data")
    train_sents, test_sents, vocab = split_and_map_unk(all_sents)

    train_tokens = sum(len(s) for s in train_sents)
    test_tokens = sum(len(s) for s in test_sents)

    print(f"Total sentences : {len(all_sents):,}")
    print(f"Train tokens    : {train_tokens:,}")
    print(f"Test tokens     : {test_tokens:,}")
    print(f"Vocab size      : {len(vocab):,}")
    print(f"Sample sentence : {train_sents[10]}")