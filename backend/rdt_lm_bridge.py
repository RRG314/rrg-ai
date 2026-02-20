from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass


def rdt_log_depth(n: int, alpha: float = 1.5) -> int:
    """
    Log-log style RDT depth adapted from rdt_lm.ipynb experiments.
    """
    if n < 2:
        return 0
    x = int(n)
    depth = 0
    while x > 1 and depth < 1000:
        d = max(2, int(math.log(max(2, x)) ** float(alpha)))
        if x < d:
            break
        x //= d
        depth += 1
    return int(depth)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_'-]*", text or "")]


@dataclass(slots=True)
class ShellStats:
    alpha: float
    token_count: int
    unique_count: int
    shell_histogram: dict[int, int]


class RDTNgramGenerator:
    """
    Lightweight shell-aware n-gram generator based on notebook logic,
    implemented without heavy dependencies.
    """

    def __init__(self, alpha: float = 1.5, seed: int = 42) -> None:
        self.alpha = float(alpha)
        self.random = random.Random(seed)
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.word_depth: dict[str, int] = {}
        self.shells: dict[int, list[str]] = defaultdict(list)
        self.bigrams: dict[str, Counter[str]] = defaultdict(Counter)
        self.trigrams: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.fitted = False

    def fit(self, sentences: list[str] | list[list[str]]) -> None:
        rows: list[list[str]] = []
        for item in sentences:
            if isinstance(item, list):
                toks = [str(x).lower() for x in item if str(x).strip()]
            else:
                toks = _tokenize(str(item))
            if toks:
                rows.append(toks)

        vocab = sorted({w for sent in rows for w in sent})
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        for word in vocab:
            self.word2idx[word] = len(self.word2idx)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        self.word_depth = {}
        self.shells = defaultdict(list)
        for word, idx in self.word2idx.items():
            if idx < 2:
                continue
            depth = rdt_log_depth(idx, alpha=self.alpha)
            self.word_depth[word] = depth
            self.shells[depth].append(word)

        self.bigrams = defaultdict(Counter)
        self.trigrams = defaultdict(Counter)
        for sent in rows:
            for i in range(len(sent) - 1):
                self.bigrams[sent[i]][sent[i + 1]] += 1
            for i in range(len(sent) - 2):
                self.trigrams[(sent[i], sent[i + 1])][sent[i + 2]] += 1

        self.fitted = True

    def generate(self, start_text: str, max_length: int = 15) -> str:
        if not self.fitted:
            self.fit(default_rdt_sentences())

        words = _tokenize(start_text)
        if not words:
            words = ["the"]
        generated = list(words)
        recent: set[str] = set(words[-3:])

        for _ in range(max_length):
            last = generated[-1]
            if last not in self.word2idx:
                break

            if len(generated) >= 2:
                key = (generated[-2], generated[-1])
                trig = self.trigrams.get(key)
                if trig:
                    choices = [(w, c) for w, c in trig.items() if w not in recent]
                    if choices:
                        nxt = self.random.choices([x[0] for x in choices], weights=[x[1] for x in choices], k=1)[0]
                        generated.append(nxt)
                        recent.add(nxt)
                        if len(recent) > 6:
                            recent.pop()
                        continue

            scores: dict[str, float] = {}
            big = self.bigrams.get(last)
            if big:
                total = sum(big.values())
                if total > 0:
                    for w, c in big.items():
                        if w in recent:
                            continue
                        scores[w] = (c / total) * 0.6

            last_shell = self.word_depth.get(last, 0)
            for off in (0, 1, -1):
                shell = last_shell + off
                for w in self.shells.get(shell, []):
                    if w == last or w in recent:
                        continue
                    base = 0.15
                    if self.word_depth.get(w, 0) == last_shell:
                        base *= 1.4
                    scores[w] = scores.get(w, 0.0) + base

            if not scores:
                break

            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            topk = sorted_words[:3]
            if not topk:
                break
            nxt = self.random.choice(topk)[0]
            generated.append(nxt)
            recent.add(nxt)
            if len(recent) > 6:
                recent.pop()

            if len(generated) >= 4 and len(set(generated[-4:])) <= 2:
                break

        return " ".join(generated)

    def shell_stats(self, text: str) -> ShellStats:
        toks = _tokenize(text)
        if not self.fitted:
            self.fit(default_rdt_sentences())

        hist: dict[int, int] = defaultdict(int)
        for tok in toks:
            idx = self.word2idx.get(tok, 1)
            depth = rdt_log_depth(idx, alpha=self.alpha)
            hist[depth] += 1

        return ShellStats(
            alpha=self.alpha,
            token_count=len(toks),
            unique_count=len(set(toks)),
            shell_histogram={k: hist[k] for k in sorted(hist)},
        )


def shell_alignment_score(query: str, snippets: list[str], alpha: float = 1.5) -> dict[str, float | int]:
    query_tokens = [t for t in _tokenize(query) if len(t) >= 2]
    snippet_tokens = [t for s in snippets for t in _tokenize(s) if len(t) >= 2]

    if not query_tokens or not snippet_tokens:
        return {
            "query_token_count": len(query_tokens),
            "snippet_token_count": len(snippet_tokens),
            "shell_overlap": 0.0,
            "token_overlap": 0.0,
        }

    # Build stable pseudo indices from token frequency order.
    q_vocab = sorted(set(query_tokens))
    s_vocab = sorted(set(snippet_tokens))
    q_index = {w: i + 2 for i, w in enumerate(q_vocab)}
    s_index = {w: i + 2 for i, w in enumerate(s_vocab)}

    q_shells = {rdt_log_depth(q_index[w], alpha=alpha) for w in q_vocab}
    s_shells = {rdt_log_depth(s_index[w], alpha=alpha) for w in s_vocab}

    shell_overlap = len(q_shells.intersection(s_shells)) / max(1, len(q_shells.union(s_shells)))

    q_set = set(query_tokens)
    s_set = set(snippet_tokens)
    token_overlap = len(q_set.intersection(s_set)) / max(1, len(q_set.union(s_set)))

    return {
        "query_token_count": len(query_tokens),
        "snippet_token_count": len(snippet_tokens),
        "shell_overlap": float(shell_overlap),
        "token_overlap": float(token_overlap),
    }


def default_rdt_sentences() -> list[str]:
    return [
        "the man walked to the house",
        "the woman went to the market",
        "the child ran to the river",
        "he was happy and she was sad",
        "they found the truth and left quickly",
        "in the morning he woke up early",
        "the old king ruled the land wisely",
        "once there was a wise teacher",
        "the forest was dark and very quiet",
        "the young prince sought great adventure",
    ]
