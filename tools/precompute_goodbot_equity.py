"""Precompute a small equity table for GoodBot.

Lets GoodBot skip Monte Carlo in the most common spots. It auto-loads
goodbot_precomputed_equity.pkl.gz from opponent_pool/ if present, or whatever
GOODBOT_PRECOMPUTED_EQUITY points at.

Keys are (my_key, board_key, bias_bucket): sorted card tuples plus the opponent
range bias, bucketed to a 0.1 grid by default (--bias-buckets "0.0:1.0:0.05"
for finer). Runs resume: an existing --out is loaded and only missing keys are
computed.

    python tools/precompute_goodbot_equity.py --mode preflop_exact
    python tools/precompute_goodbot_equity.py --mode texture
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
import pickle
import random
import time
from itertools import combinations
from collections import Counter

import pkrbot

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
FULL_DECK = [r + s for r in RANKS for s in SUITS]

_RANK_TO_INT = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
}


def _card_rank_char(code: str) -> str:
    s = str(code).strip()
    if s.startswith('10'):
        return 'T'
    return s[0].upper() if s else '2'


def _card_suit_char(code: str) -> str:
    s = str(code).strip()
    if s.startswith('10') and len(s) >= 3:
        return s[2].lower()
    return s[1].lower() if len(s) >= 2 else ''


def _are_suited(c1: str, c2: str) -> bool:
    s1 = _card_suit_char(c1)
    s2 = _card_suit_char(c2)
    return bool(s1) and bool(s2) and s1 == s2


def _hand_strength_proxy(c1: str, c2: str) -> float:
    # Chen-style preflop scoring mapped to 0..1.
    r1 = _RANK_TO_INT.get(_card_rank_char(c1), 2)
    r2 = _RANK_TO_INT.get(_card_rank_char(c2), 2)
    hi = max(r1, r2)
    lo = min(r1, r2)
    pair = (r1 == r2)
    suited = _are_suited(c1, c2)
    gap = hi - lo

    base_map = {
        14: 10.0,
        13: 8.0,
        12: 7.0,
        11: 6.0,
        10: 5.0,
        9: 4.5,
        8: 4.0,
        7: 3.5,
        6: 3.0,
        5: 2.5,
        4: 2.0,
        3: 1.5,
        2: 1.0,
    }
    score = float(base_map.get(hi, 1.0))

    if pair:
        score *= 2.0
        if score < 5.0:
            score = 5.0
    else:
        if suited:
            score += 2.0

        if gap == 1:
            score -= 1.0
        elif gap == 2:
            score -= 2.0
        elif gap == 3:
            score -= 4.0
        elif gap == 4:
            score -= 5.0
        elif gap >= 5:
            score -= 6.0

        if gap <= 1 and hi <= 11:
            score += 1.0

        if lo <= 5 and hi <= 11 and not suited:
            score -= 0.5

    score = max(0.0, min(20.0, score))
    x = (score - 8.0) / 2.5
    s = 1.0 / (1.0 + math.exp(-x))
    return max(0.0, min(1.0, float(s)))


def _board_texture_key(board: tuple[str, ...]) -> tuple[int, ...]:
    """Lossy board descriptor: discretized so the table gets hits during play
    instead of exploding to exact-board size."""
    if not board:
        return (0, 0, 0, 0, 0, 0, 0)

    ranks = [_RANK_TO_INT.get(_card_rank_char(c), 2) for c in board]
    suits = [_card_suit_char(c) for c in board]

    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)

    counts_sorted = sorted(rank_counts.values(), reverse=True)
    top1 = counts_sorted[0] if counts_sorted else 0
    top2 = counts_sorted[1] if len(counts_sorted) > 1 else 0

    num_pairs = sum(1 for v in rank_counts.values() if v == 2)
    has_trips = 1 if any(v == 3 for v in rank_counts.values()) else 0
    has_quads = 1 if any(v >= 4 for v in rank_counts.values()) else 0

    suit_sorted = sorted(suit_counts.values(), reverse=True)
    max_suit = suit_sorted[0] if suit_sorted else 0
    second_suit = suit_sorted[1] if len(suit_sorted) > 1 else 0

    uniq = sorted(set(ranks))
    if 14 in uniq:
        uniq = sorted(set(uniq + [1]))  # wheel-ish handling
    best_run = 1
    cur = 1
    for i in range(1, len(uniq)):
        if uniq[i] == uniq[i - 1] + 1:
            cur += 1
            best_run = max(best_run, cur)
        else:
            cur = 1

    high = max(ranks) if ranks else 0
    low = min(ranks) if ranks else 0
    spread = max(0, high - low)
    distinct = len(rank_counts)

    broadways = sum(1 for r in ranks if r >= 11)
    lows = sum(1 for r in ranks if r <= 5)
    has_ace = 1 if 14 in ranks else 0
    suitedness_class = 0
    # 0=rnbw-ish, 1=two-tone, 2=three-tone+, 3=mono-ish
    if max_suit >= 4:
        suitedness_class = 3
    elif max_suit == 3:
        suitedness_class = 2
    elif max_suit == 2:
        suitedness_class = 1

    # Bucket some high-card / spread values for more compactness.
    high_bucket = 0
    if high >= 13:
        high_bucket = 3
    elif high >= 11:
        high_bucket = 2
    elif high >= 9:
        high_bucket = 1

    spread_bucket = 0
    if spread >= 10:
        spread_bucket = 3
    elif spread >= 7:
        spread_bucket = 2
    elif spread >= 4:
        spread_bucket = 1

    return (
        distinct,
        top1,
        top2,
        num_pairs,
        has_trips,
        has_quads,
        max_suit,
        second_suit,
        suitedness_class,
        best_run,
        high_bucket,
        spread_bucket,
        broadways,
        lows,
        has_ace,
    )


def _bias_weight(strength: float, bias: float) -> float:
    # Match GoodBot weighting: exp(2*bias*(strength-0.5))
    return math.exp(2.0 * bias * (float(strength) - 0.5))


def _build_global_opp_sampler(bias: float):
    """Weighted sampler over all 1326 2-card combos; callers reject overlaps."""
    combos: list[tuple[str, str]] = []
    cum: list[float] = []
    total = 0.0
    for i in range(len(FULL_DECK)):
        c1 = FULL_DECK[i]
        for j in range(i + 1, len(FULL_DECK)):
            c2 = FULL_DECK[j]
            strength = _hand_strength_proxy(c1, c2)
            w = _bias_weight(strength, bias)
            total += w
            combos.append((c1, c2))
            cum.append(total)
    return combos, cum, total


def _sample_opp_from_global(global_sampler, used: set[str]) -> tuple[str, str]:
    combos, cum, total = global_sampler
    # Rejection sampling for overlap.
    for _ in range(64):
        r = random.random() * total
        # bisect without importing bisect (keep deps minimal)
        lo = 0
        hi = len(cum)
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        idx = lo if lo < len(combos) else (len(combos) - 1)
        c1, c2 = combos[idx]
        if c1 not in used and c2 not in used:
            return c1, c2
    # Uniform fallback
    pool = [c for c in FULL_DECK if c not in used]
    c1, c2 = random.sample(pool, 2)
    return c1, c2


def _equity_with_board(my_key: tuple[str, str], board_key: tuple[str, ...], iters: int, bias: float, card_objs, global_sampler) -> float:
    used_base = set(my_key) | set(board_key)
    my_hand = [card_objs[my_key[0]], card_objs[my_key[1]]]
    board_cards = [card_objs[c] for c in board_key]
    remaining = 6 - len(board_key)

    wins = 0
    ties = 0
    for _ in range(max(1, int(iters))):
        used = set(used_base)
        o1, o2 = _sample_opp_from_global(global_sampler, used)
        used.add(o1)
        used.add(o2)

        if remaining > 0:
            pool = [c for c in FULL_DECK if c not in used]
            fill = random.sample(pool, remaining)
            full_board = board_cards + [card_objs[c] for c in fill]
        else:
            full_board = board_cards

        opp_hand = [card_objs[o1], card_objs[o2]]
        my_score = pkrbot.evaluate(full_board + my_hand)
        opp_score = pkrbot.evaluate(full_board + opp_hand)
        if my_score > opp_score:
            wins += 1
        elif my_score == opp_score:
            ties += 1

    return (wins + 0.5 * ties) / max(1, int(iters))


def _build_weighted_opp_sampler(remaining_cards: list[str], bias: float):
    """Return (combos, cum_weights, total_weight) for opponent sampling."""
    combos: list[tuple[str, str]] = []
    cum: list[float] = []
    total = 0.0
    n = len(remaining_cards)

    for i in range(n):
        c1 = remaining_cards[i]
        for j in range(i + 1, n):
            c2 = remaining_cards[j]
            strength = _hand_strength_proxy(c1, c2)
            w = math.exp(2.0 * bias * (strength - 0.5))
            total += w
            combos.append((c1, c2))
            cum.append(total)

    return combos, cum, total


def _sample_opp(combos, cum, total) -> tuple[str, str]:
    r = random.random() * total
    # manual bisect to avoid importing bisect in tight loop
    lo, hi = 0, len(cum)
    while lo < hi:
        mid = (lo + hi) // 2
        if cum[mid] < r:
            lo = mid + 1
        else:
            hi = mid
    idx = lo if lo < len(combos) else (len(combos) - 1)
    return combos[idx]


def _equity_preflop(my2: tuple[str, str], iters: int, bias: float, card_objs: dict[str, pkrbot.Card]) -> float:
    dead = set(my2)
    remaining = [c for c in FULL_DECK if c not in dead]

    combos = cum = None
    total = 0.0
    if bias > 0.0:
        combos, cum, total = _build_weighted_opp_sampler(remaining, bias)

    wins = 0
    ties = 0

    my_hand = [card_objs[my2[0]], card_objs[my2[1]]]

    for _ in range(iters):
        if bias > 0.0 and combos and total > 0.0:
            o1, o2 = _sample_opp(combos, cum, total)
        else:
            o1, o2 = random.sample(remaining, 2)

        board_pool = [c for c in remaining if c != o1 and c != o2]
        board6 = random.sample(board_pool, 6)

        board_cards = [card_objs[c] for c in board6]
        opp_hand = [card_objs[o1], card_objs[o2]]

        my_score = pkrbot.evaluate(board_cards + my_hand)
        opp_score = pkrbot.evaluate(board_cards + opp_hand)
        if my_score > opp_score:
            wins += 1
        elif my_score == opp_score:
            ties += 1

    return (wins + 0.5 * ties) / max(1, iters)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['preflop_exact', 'texture'], default='preflop_exact')

    default_out_exact = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'opponent_pool', 'goodbot_precomputed_equity.pkl.gz')
    )
    default_out_texture = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'opponent_pool', 'goodbot_precomputed_equity_texture.pkl.gz')
    )
    ap.add_argument('--out', default='')

    ap.add_argument('--iters', type=int, default=10000,
                    help='Rollouts per state (accuracy knob; does not increase file size much).')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--bias-buckets', default='0.0:1.0:0.01')
    ap.add_argument('--checkpoint-every', type=int, default=250)
    ap.add_argument('--texture-samples', type=int, default=50000,
                    help='Number of random (hand,board) states per board length per bias bucket in texture mode.')
    ap.add_argument('--texture-board-lens', default='2,3,4,5,6',
                    help='Comma-separated board lengths to include in texture mode (recommended: 2,3,4,5,6).')
    ap.add_argument('--hand-buckets', type=int, default=100,
                    help='Number of hand-strength buckets in texture mode (more buckets => larger table).')
    args = ap.parse_args()

    random.seed(args.seed)

    def _parse_bias_buckets(spec: str) -> list[float]:
        spec = str(spec or '').strip()
        if not spec:
            return [0.0]

        buckets: list[float] = []

        # Range format: start:end:step
        if ':' in spec and ',' not in spec:
            parts = [p.strip() for p in spec.split(':')]
            if len(parts) != 3:
                raise ValueError('bias range must be start:end:step')
            start = float(parts[0])
            end = float(parts[1])
            step = float(parts[2])
            if step <= 0:
                raise ValueError('bias step must be > 0')
            start = max(0.0, min(1.0, start))
            end = max(0.0, min(1.0, end))
            if end < start:
                start, end = end, start
            k = 0
            while True:
                b = start + k * step
                if b > end + 1e-9:
                    break
                b = max(0.0, min(1.0, b))
                # Keep the requested granularity; round for float stability.
                b = round(b, 4)
                if b not in buckets:
                    buckets.append(b)
                k += 1
        else:
            # Comma format
            for s in spec.split(','):
                s = s.strip()
                if not s:
                    continue
                b = float(s)
                b = max(0.0, min(1.0, b))
                b = round(b, 4)
                if b not in buckets:
                    buckets.append(b)

        buckets.sort()
        return buckets

    bias_buckets = _parse_bias_buckets(args.bias_buckets)

    card_objs = {c: pkrbot.Card(c) for c in FULL_DECK}

    table = {}
    out_path = os.path.normpath(str(args.out).strip())
    if not out_path:
        out_path = default_out_exact if args.mode == 'preflop_exact' else default_out_texture

    # A directory arg means "write the default name in here", not "overwrite me".
    if os.path.isdir(out_path):
        out_path = os.path.join(
            out_path,
            'goodbot_precomputed_equity.pkl.gz' if args.mode == 'preflop_exact' else 'goodbot_precomputed_equity_texture.pkl.gz'
        )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_path):
        try:
            with gzip.open(out_path, 'rb') as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict):
                table = loaded
                print(f"Resuming from {out_path} with {len(table)} existing entries")
        except Exception:
            table = {}
    if args.mode == 'preflop_exact':
        board_key: tuple[str, ...] = tuple()  # empty board
        hands = list(combinations(FULL_DECK, 2))
        total_jobs = len(hands) * len(bias_buckets)

        missing = 0
        for my2 in hands:
            my_key = tuple(sorted(my2))
            for b in bias_buckets:
                if (my_key, board_key, b) not in table:
                    missing += 1

        done = 0
        for my2 in hands:
            my_key = tuple(sorted(my2))
            for b in bias_buckets:
                state_key = (my_key, board_key, b)
                if state_key not in table:
                    eq = _equity_preflop(my_key, iters=max(1, int(args.iters)), bias=b, card_objs=card_objs)
                    table[state_key] = float(eq)

                done += 1
                if done % 250 == 0:
                    print(f"{done}/{total_jobs} looped; table size {len(table)}")
                if int(args.checkpoint_every) > 0 and (done % int(args.checkpoint_every) == 0):
                    # Best-effort checkpoint; tolerate transient Windows file locks.
                    tmp = out_path + f".tmp.{os.getpid()}"
                    with gzip.open(tmp, 'wb') as f:
                        pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)
                    for attempt in range(25):
                        try:
                            os.replace(tmp, out_path)
                            break
                        except OSError:
                            if attempt == 24:
                                print(f"WARNING: could not replace {out_path}; left temp file {tmp}")
                            else:
                                time.sleep(0.1)
    else:
        # Texture mode: accumulate equity by (hand_strength_bucket, board_texture, board_len, bias_bucket).
        board_lens: list[int] = []
        for s in str(args.texture_board_lens).split(','):
            s = s.strip()
            if not s:
                continue
            try:
                board_lens.append(int(s))
            except Exception:
                pass
        board_lens = [L for L in board_lens if 0 <= L <= 6]
        if not board_lens:
            board_lens = [2, 4, 5]

        hand_buckets = max(2, int(args.hand_buckets))
        samples = max(1, int(args.texture_samples))

        sums: dict[tuple, float] = {}
        counts: dict[tuple, int] = {}

        for b in bias_buckets:
            global_sampler = _build_global_opp_sampler(float(b))
            for L in board_lens:
                for n in range(samples):
                    my2 = tuple(sorted(random.sample(FULL_DECK, 2)))
                    used = set(my2)
                    pool = [c for c in FULL_DECK if c not in used]
                    board = tuple(sorted(random.sample(pool, L)))

                    strength = _hand_strength_proxy(my2[0], my2[1])
                    hb = int(max(0, min(hand_buckets - 1, math.floor(float(strength) * hand_buckets))))
                    tex = _board_texture_key(board)

                    eq = _equity_with_board(my2, board, iters=max(1, int(args.iters)), bias=float(b), card_objs=card_objs, global_sampler=global_sampler)
                    key = ('tex', hb, L, tex, float(b))
                    sums[key] = sums.get(key, 0.0) + float(eq)
                    counts[key] = counts.get(key, 0) + 1

                    if (n + 1) % 250 == 0:
                        print(f"bias={b} L={L} {n+1}/{samples} samples; table keys {len(sums)}")
                    if int(args.checkpoint_every) > 0 and ((n + 1) % int(args.checkpoint_every) == 0):
                        for k2, s2 in sums.items():
                            c2 = counts.get(k2, 1)
                            table[k2] = float(s2) / max(1, int(c2))
                        tmp = out_path + f".tmp.{os.getpid()}"
                        with gzip.open(tmp, 'wb') as f:
                            pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)
                        for attempt in range(25):
                            try:
                                os.replace(tmp, out_path)
                                break
                            except OSError:
                                if attempt == 24:
                                    print(f"WARNING: could not replace {out_path}; left temp file {tmp}")
                                else:
                                    time.sleep(0.1)

        for k2, s2 in sums.items():
            c2 = counts.get(k2, 1)
            table[k2] = float(s2) / max(1, int(c2))

    def _checkpoint():
        tmp = out_path + f".tmp.{os.getpid()}"
        with gzip.open(tmp, 'wb') as f:
            pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)
        for attempt in range(25):
            try:
                os.replace(tmp, out_path)
                return
            except OSError:
                if attempt == 24:
                    print(f"WARNING: could not replace {out_path}; left temp file {tmp}")
                    return
                time.sleep(0.1)

    _checkpoint()

    print(f"Wrote {len(table)} entries -> {out_path}")
    print(
        "If this file is in opponent_pool/ with the default name, "
        "GoodBot will load it automatically. Otherwise set GOODBOT_PRECOMPUTED_EQUITY to this path."
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
