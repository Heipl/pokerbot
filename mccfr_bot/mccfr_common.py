import math
import pkrbot

MCCFR_ACTIONS = (
    "fold",
    "call_check",
    "raise_xsmall",
    "raise_small",
    "raise_med",
    "raise_large",
    "raise_over",
    "raise_allin",
    "discard_0",
    "discard_1",
    "discard_2",
)

MCCFR_ALLOWED_RAISES = (3, 4, 6, 7)

MCCFR_RAISE_FRACTIONS = {
    "default": {2: 0.30, 3: 0.50, 4: 1.00, 5: 1.50, 6: 2.00},
    4: {2: 0.25, 3: 0.40, 4: 0.85, 5: 1.25, 6: 1.80},
    5: {2: 0.30, 3: 0.60, 4: 1.10, 5: 1.60, 6: 2.10},
}

POT_BB_BUCKETS = {
    "default": [3, 8, 16, 32],
    4: [3, 7, 14, 28],
    5: [4, 9, 18, 36],
}
SPR_BUCKETS = {
    "default": [1.5, 4, 9],
    4: [1.0, 2.5, 6],
    5: [0.8, 2.0, 5],
}
PRESSURE_BUCKETS = {
    "default": [0.2, 0.45, 0.7],
    4: [0.12, 0.30, 0.55],
    5: [0.15, 0.35, 0.60],
}
HAND_BUCKETS = {"default": 4, 4: 8}
BOARD_BUCKETS = {"default": 16, 4: 32}

_RANK_CHARS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
_RANK_TO_INT = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}

def _card_rank_char(card) -> str:
    try:
        s = str(card).strip()
        if s.startswith("10"):
            return "T"
        if s and s[0].upper() in _RANK_TO_INT:
            return s[0].upper()
    except Exception:
        pass
    try:
        return _RANK_CHARS[int(getattr(card, "rank"))]
    except Exception:
        return "2"

def _card_suit_char(card) -> str:
    try:
        s = str(card).strip()
        if s.startswith("10") and len(s) >= 3:
            return s[2].lower()
        return s[1].lower() if len(s) >= 2 else ""
    except Exception:
        return ""

def _are_suited(c1, c2) -> bool:
    s1 = _card_suit_char(c1)
    s2 = _card_suit_char(c2)
    return bool(s1) and bool(s2) and s1 == s2

def _board_texture_key(board_cards) -> tuple:
    try:
        codes = [str(c) for c in (board_cards or [])]
    except Exception:
        codes = []
    if not codes:
        return (0, 0, 0, 0, 0, 0, 0)

    ranks = [_RANK_TO_INT.get(_card_rank_char(c), 2) for c in codes]
    suits = [_card_suit_char(c) for c in codes]

    try:
        from collections import Counter
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
    except Exception:
        rank_counts = {}
        suit_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1

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
        uniq = sorted(set(uniq + [1]))
    best_run = 1
    cur = 1
    for i in range(1, len(uniq)):
        if uniq[i] == uniq[i - 1] + 1:
            cur += 1
            if cur > best_run:
                best_run = cur
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
    if max_suit >= 4:
        suitedness_class = 3
    elif max_suit == 3:
        suitedness_class = 2
    elif max_suit == 2:
        suitedness_class = 1

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

def _hash_bucket(values, mod):
    h = 0
    for v in values:
        h = (h * 131 + int(v) + 7) % mod
    return int(h)

def board_texture_bucket(board_cards, bucket_count=BOARD_BUCKETS) -> int:
    if isinstance(bucket_count, dict):
        bucket_count = select_bucket_count(bucket_count, 0, 32)
    try:
        bucket_count = max(1, int(bucket_count))
    except Exception:
        bucket_count = 32
    tex = _board_texture_key(board_cards)
    if not tex or len(tex) < 12:
        return 0
    suited = max(0, min(3, int(tex[8])))
    top1 = int(tex[1])
    paired3 = 0 if top1 <= 1 else (1 if top1 == 2 else 2)
    paired2 = 0 if top1 <= 1 else 1
    run = int(tex[9])
    connect3 = 0 if run <= 2 else (1 if run == 3 else 2)
    connect2 = 0 if run <= 3 else 1
    high4 = max(0, min(3, int(tex[10])))
    high2 = 0 if high4 <= 1 else 1

    if bucket_count >= 144:
        idx = ((suited * 3 + paired3) * 3 + connect3) * 4 + high4
    elif bucket_count >= 48:
        idx = ((suited * 3 + paired3) * 2 + connect2) * 2 + high2
    elif bucket_count >= 32:
        idx = ((suited * 2 + paired2) * 2 + connect2) * 2 + high2
    elif bucket_count >= 16:
        idx = (suited * 2 + paired2) * 2 + connect2
    elif bucket_count >= 8:
        idx = suited * 2 + paired2
    else:
        idx = suited
    return int(min(idx, bucket_count - 1))

def _chen_score(c1, c2) -> float:
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
    return max(0.0, min(20.0, score))

def _to_card(card):
    if isinstance(card, pkrbot.Card):
        return card
    return pkrbot.Card(str(card))

def hand_strength_bucket(cards, board, score_ranges, bucket_count=HAND_BUCKETS) -> int:
    if isinstance(bucket_count, dict):
        bucket_count = select_bucket_count(bucket_count, 0, 8)
    try:
        bucket_count = int(bucket_count)
    except Exception:
        bucket_count = 8
    cards = list(cards or [])
    board = list(board or [])
    if len(cards) == 3:
        best = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                best = max(best, _chen_score(cards[i], cards[j]))
        return int(min(bucket_count - 1, max(0, (best / 20.0) * bucket_count)))
    if len(cards) != 2:
        return bucket_count // 2
    if len(board) < 3:
        score = _chen_score(cards[0], cards[1])
        return int(min(bucket_count - 1, max(0, (score / 20.0) * bucket_count)))
    try:
        score = int(pkrbot.evaluate([_to_card(c) for c in (board + cards)]))
        rng = score_ranges.get(len(board))
        if isinstance(rng, (list, tuple)) and len(rng) > 2:
            cuts = list(rng)
            lo, hi = 0, len(cuts)
            while lo < hi:
                mid = (lo + hi) // 2
                if score <= cuts[mid]:
                    hi = mid
                else:
                    lo = mid + 1
            frac = lo / float(len(cuts) + 1)
            return int(min(bucket_count - 1, max(0, frac * bucket_count)))
        min_s, max_s = (rng if isinstance(rng, (list, tuple)) and len(rng) == 2
                        else (None, None))
        if min_s is None or max_s is None or max_s <= min_s:
            norm = min(1.0, max(0.0, (score - 1_000_000) / 9_000_000))
        else:
            norm = (score - min_s) / float(max_s - min_s)
        return int(min(bucket_count - 1, max(0, norm * bucket_count)))
    except Exception:
        return bucket_count // 2

def three_card_shape(cards) -> tuple:
    cards = list(cards or [])
    if not cards:
        return (0, 0, 0)
    ranks = sorted([_RANK_TO_INT.get(_card_rank_char(c), 2) for c in cards], reverse=True)
    suits = [_card_suit_char(c) for c in cards]
    suit_counts = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    suited_class = 0
    if suit_counts and max(suit_counts.values()) >= 3:
        suited_class = 2
    elif suit_counts and max(suit_counts.values()) == 2:
        suited_class = 1
    pair_class = 0
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    max_count = max(rank_counts.values()) if rank_counts else 1
    if max_count >= 3:
        pair_class = 2
    elif max_count == 2:
        pair_class = 1
    gap = max(ranks) - min(ranks) if ranks else 0
    gap_bucket = bucketize(gap, [4, 7, 10])
    return (pair_class, suited_class, gap_bucket)

def bucketize(value: float, edges) -> int:
    for i, e in enumerate(edges):
        if value <= e:
            return i
    return len(edges)

def _select_bucket_list(buckets, street: int):
    if isinstance(buckets, dict):
        if street in buckets:
            return buckets[street]
        if str(street) in buckets:
            return buckets[str(street)]
        return buckets.get("default", next(iter(buckets.values())))
    return buckets

def select_bucket_count(buckets, street: int, fallback: int) -> int:
    if isinstance(buckets, dict):
        if street in buckets:
            return int(buckets[street])
        if str(street) in buckets:
            return int(buckets[str(street)])
        if "default" in buckets:
            return int(buckets["default"])
        try:
            return int(next(iter(buckets.values())))
        except Exception:
            return int(fallback)
    try:
        return int(buckets)
    except Exception:
        return int(fallback)

def in_position(street: int, active: int) -> bool:
    if street == 0:
        return active == 1
    if street >= 4:
        return active == 0
    return active == 0

def info_key(
    street: int,
    active: int,
    pot_bb: float,
    spr: float,
    pressure: float,
    facing_raise: bool,
    raise_count_bucket: int,
    last_aggressor: int,
    last_bet_bucket: int,
    hand_bucket: int,
    board_bucket: int,
    shape: tuple,
    pot_buckets=None,
    spr_buckets=None,
    pressure_buckets=None,
) -> tuple:
    pot_buckets = POT_BB_BUCKETS if pot_buckets is None else pot_buckets
    spr_buckets = SPR_BUCKETS if spr_buckets is None else spr_buckets
    pressure_buckets = PRESSURE_BUCKETS if pressure_buckets is None else pressure_buckets
    pot_bucket = bucketize(pot_bb, _select_bucket_list(pot_buckets, street))
    spr_bucket = bucketize(spr, _select_bucket_list(spr_buckets, street))
    pressure_bucket = bucketize(pressure, _select_bucket_list(pressure_buckets, street))
    return (
        int(street),
        int(active),
        int(in_position(street, active)),
        int(pot_bucket),
        int(spr_bucket),
        int(pressure_bucket),
        int(facing_raise),
        int(raise_count_bucket),
        int(last_aggressor),
        int(last_bet_bucket),
        int(hand_bucket),
        int(board_bucket),
        int(shape[0]),
        int(shape[1]),
        int(shape[2]),
    )

def get_raise_fraction(street: int, action_idx: int) -> float:
    if isinstance(MCCFR_RAISE_FRACTIONS, dict):
        frac_map = MCCFR_RAISE_FRACTIONS.get(street)
        if not isinstance(frac_map, dict):
            frac_map = MCCFR_RAISE_FRACTIONS.get("default", {})
        if isinstance(frac_map, dict):
            if action_idx in frac_map:
                return float(frac_map[action_idx])
    return 0.75
