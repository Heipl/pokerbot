"""Feature extraction for PG/PPO policies.

Shared by tools/train_goodbot_pg.py and mccfr_bot/player.py, so the feature
order here is a wire format: append new features at the end, never reorder.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import List, Tuple

RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_TO_I = {r: i for i, r in enumerate(RANKS)}
SUIT_TO_I = {s: i for i, s in enumerate(SUITS)}

def _rank_char(card) -> str:
    s = str(card).strip()
    if s.startswith("10"):
        return "T"
    return s[0].upper() if s else "2"

def _suit_char(card) -> str:
    s = str(card).strip()
    if s.startswith("10") and len(s) >= 3:
        return s[2].lower()
    return s[1].lower() if len(s) >= 2 else "s"

def encode_card_onehot(card) -> List[float]:
    r = _rank_char(card)
    s = _suit_char(card)
    v = [0.0] * (13 + 4)
    v[RANK_TO_I.get(r, 0)] = 1.0
    v[13 + SUIT_TO_I.get(s, 0)] = 1.0
    return v

def _board_texture_features(board_cards) -> List[float]:
    cards = list(board_cards or [])
    if not cards:
        return [0.0] * 10

    ranks = [RANK_TO_I.get(_rank_char(c), 0) + 2 for c in cards]
    suits = [_suit_char(c) for c in cards]

    rc = Counter(ranks)
    sc = Counter(suits)
    counts = sorted(rc.values(), reverse=True)
    top1 = counts[0] if counts else 0
    top2 = counts[1] if len(counts) > 1 else 0
    num_pairs = sum(1 for v in rc.values() if v == 2)
    has_trips = 1.0 if any(v == 3 for v in rc.values()) else 0.0
    has_quads = 1.0 if any(v >= 4 for v in rc.values()) else 0.0
    max_suit = max(sc.values()) if sc else 0

    uniq = sorted(set(ranks))
    if 14 in uniq:
        uniq = sorted(set(uniq + [1]))
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

    return [
        float(len(rc)) / 6.0,
        float(top1) / 4.0,
        float(top2) / 4.0,
        float(num_pairs) / 3.0,
        float(has_trips),
        float(has_quads),
        float(max_suit) / 6.0,
        float(best_run) / 6.0,
        float(high) / 14.0,
        float(spread) / 14.0,
    ]

def _action_history_features(round_state, active: int, big_blind: int) -> List[float]:
    prev = getattr(round_state, "previous_state", None)
    street = int(getattr(round_state, "street", 0) or 0)

    last_type = [0.0, 0.0, 0.0, 0.0, 0.0]
    last_bet_bb = 0.0
    raises = 0.0

    cur = round_state
    steps = 0
    while prev is not None and getattr(prev, "street", None) == street and steps < 12:
        try:
            h0 = len(getattr(cur, "hands")[0])
            h1 = len(getattr(cur, "hands")[1])
            ph0 = len(getattr(prev, "hands")[0])
            ph1 = len(getattr(prev, "hands")[1])
            if (h0 != ph0) or (h1 != ph1):
                last_type = [0.0, 0.0, 0.0, 0.0, 1.0]
        except Exception:
            pass

        try:
            cur_max = max(getattr(cur, "pips"))
            prev_max = max(getattr(prev, "pips"))
            if cur_max > prev_max:
                raises += 1.0
                if last_bet_bb == 0.0:
                    last_bet_bb = float(cur_max - prev_max) / float(max(1, big_blind))
                    last_type = [0.0, 0.0, 0.0, 1.0, last_type[4]]
        except Exception:
            pass

        cur = prev
        prev = getattr(prev, "previous_state", None)
        steps += 1

    if sum(last_type) == 0.0:
        try:
            continue_cost = int(round_state.pips[1 - active] - round_state.pips[active])
            if continue_cost > 0:
                last_type = [0.0, 1.0, 0.0, 0.0, 0.0]
            else:
                last_type = [0.0, 0.0, 1.0, 0.0, 0.0]
        except Exception:
            last_type = [0.0, 0.0, 1.0, 0.0, 0.0]

    return last_type + [float(last_bet_bb) / 50.0, float(raises) / 6.0]

def _hand_strength_features(round_state, active: int) -> List[float]:
    try:
        from mccfr_common import _chen_score, _to_card
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]

    try:
        hole = list(round_state.hands[active] or [])
    except Exception:
        hole = []
    board = list(getattr(round_state, "board", []) or [])

    chen_norm = 0.0
    try:
        if len(hole) >= 2:
            best = 0.0
            for i in range(len(hole)):
                for j in range(i + 1, len(hole)):
                    best = max(best, float(_chen_score(hole[i], hole[j])))
            chen_norm = max(0.0, min(1.0, best / 20.0))
    except Exception:
        chen_norm = 0.0

    made_norm = 0.0
    has_made = 0.0
    try:
        if len(board) >= 3 and len(hole) >= 2:
            import pkrbot
            import itertools as _it
            _b = [_to_card(c) for c in board]
            _h = [_to_card(c) for c in hole]
            if len(_h) > 2:
                score = max(int(pkrbot.evaluate(_b + list(_c)))
                            for _c in _it.combinations(_h, 2))
            else:
                score = int(pkrbot.evaluate(_b + _h))
            made_norm = max(0.0, min(1.0, (score - 1_000_000) / 9_000_000.0))
            has_made = 1.0
    except Exception:
        made_norm = 0.0
        has_made = 0.0

    return [chen_norm, made_norm, has_made, min(1.0, len(board) / 6.0)]

def feature_size() -> int:
    return 8 + (3 * 17) + (6 * 17) + 10 + 7 + 6 + 4

def extract_features(round_state, active: int, starting_stack: int, big_blind: int, feature_dim: int = 0) -> List[float]:
    street = int(getattr(round_state, "street", 0) or 0)
    button = int(getattr(round_state, "button", 0) or 0)

    my_pip = int(round_state.pips[active])
    opp_pip = int(round_state.pips[1 - active])
    my_stack = int(round_state.stacks[active])
    opp_stack = int(round_state.stacks[1 - active])
    continue_cost = int(opp_pip - my_pip)

    my_contribution = int(starting_stack - my_stack)
    opp_contribution = int(starting_stack - opp_stack)
    pot = float(my_contribution + opp_contribution)

    st = max(0, min(5, street))
    street_oh = [0.0] * 6
    street_oh[int(st)] = 1.0

    denom_stack = float(max(1, starting_stack))
    pot_squashed = math.tanh(pot / denom_stack)
    cc_squashed = math.tanh(float(continue_cost) / denom_stack)
    scalars = [
        1.0,
        float(active),
        float(button % 2),
        float(street) / 6.0,
        float(my_stack) / denom_stack,
        float(opp_stack) / denom_stack,
        float(pot_squashed),
        float(cc_squashed),
    ]

    hole = list(round_state.hands[active] or [])
    hole_enc: List[float] = []
    for i in range(3):
        if i < len(hole):
            hole_enc.extend(encode_card_onehot(hole[i]))
        else:
            hole_enc.extend([0.0] * 17)

    board = list(getattr(round_state, "board", []) or [])
    board_enc: List[float] = []
    for i in range(6):
        if i < len(board):
            board_enc.extend(encode_card_onehot(board[i]))
        else:
            board_enc.extend([0.0] * 17)

    tex = _board_texture_features(board)
    hist = _action_history_features(round_state, active, big_blind)

    feats = scalars + hole_enc + board_enc + tex + hist + street_oh

    if not (feature_dim and 0 < int(feature_dim) <= len(feats)):
        feats = feats + _hand_strength_features(round_state, active)

    if feature_dim and feature_dim > 0:
        if len(feats) < feature_dim:
            feats.extend([0.0] * (feature_dim - len(feats)))
        elif len(feats) > feature_dim:
            feats = feats[:feature_dim]
    return feats
