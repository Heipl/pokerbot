import os, sys
here = os.path.dirname(__file__)
if here not in sys.path:
    sys.path.insert(0, here)
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import STARTING_STACK, BIG_BLIND, SMALL_BLIND, NUM_ROUNDS
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
if not hasattr(RoundState, 'get_bounty_hits'):
    def _no_bounty_hits(self):
        return (False, False)
    setattr(RoundState, 'get_bounty_hits', _no_bounty_hits)
import os
import random
import pkrbot
import math
import bisect
from itertools import combinations
import pickle
import gzip
import json
import importlib.util
from pathlib import Path
from typing import Union
from collections import deque
import numpy as np
from mccfr_common import (
    MCCFR_ACTIONS,
    MCCFR_ALLOWED_RAISES,
    MCCFR_RAISE_FRACTIONS,
    POT_BB_BUCKETS,
    SPR_BUCKETS,
    PRESSURE_BUCKETS,
    HAND_BUCKETS,
    BOARD_BUCKETS,
    board_texture_bucket,
    hand_strength_bucket,
    info_key as mccfr_info_key,
    three_card_shape,
    get_raise_fraction,
    select_bucket_count,
)

_RANK_CHARS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
_RANK_TO_INT = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

MCCFR_FOLD = 0
MCCFR_CALL_CHECK = 1
MCCFR_RAISE_XSMALL = 2
MCCFR_RAISE_SMALL = 3
MCCFR_RAISE_MED = 4
MCCFR_RAISE_LARGE = 5
MCCFR_RAISE_OVER = 6
MCCFR_RAISE_ALLIN = 7
MCCFR_DISCARD_0 = 8
MCCFR_DISCARD_1 = 9
MCCFR_DISCARD_2 = 10

def _card_rank_char(card) -> str:
    try:
        s = str(card).strip()
        if s.startswith('10'):
            return 'T'
        if s and s[0].upper() in _RANK_TO_INT:
            return s[0].upper()
    except Exception:
        pass
    try:
        return _RANK_CHARS[int(getattr(card, 'rank'))]
    except Exception:
        return '2'

def _card_suit_char(card) -> str:
    try:
        s = str(card).strip()
        if s.startswith('10') and len(s) >= 3:
            return s[2].lower()
        return s[1].lower() if len(s) >= 2 else ''
    except Exception:
        return ''

def _are_suited(c1, c2) -> bool:
    s1 = _card_suit_char(c1)
    s2 = _card_suit_char(c2)
    return bool(s1) and bool(s2) and s1 == s2

def _two_card_combos(cards):
    if len(cards) <= 2:
        yield tuple(cards)
        return
    yield (cards[0], cards[1])
    yield (cards[0], cards[2])
    yield (cards[1], cards[2])

def _board_texture_key_legacy(board_cards) -> tuple:
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
    max_suit = max(suit_counts.values()) if suit_counts else 0

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
    distinct = len(rank_counts)
    return (distinct, top1, top2, num_pairs, has_trips, has_quads, max_suit, best_run, high)

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

def _mccfr_expected_key_arity() -> int:
    return len(mccfr_info_key(0, 0, 0.0, 0.0, 0.0, False, 0, -1, 0, 0, 0, (0, 0, 0)))

def _mccfr_policy_schema_ok(policy, source: str) -> bool:
    if not policy:
        return False
    try:
        expected = _mccfr_expected_key_arity()
        for key in policy:
            if not isinstance(key, tuple):
                break
            if len(key) != expected:
                print(
                    f"[mccfr] IGNORING {source}: info keys have {len(key)} fields but "
                    f"mccfr_common now produces {expected}. The checkpoint predates the "
                    f"current schema and would miss on every lookup -- retrain with "
                    f"tools/train_mccfr.py.",
                    flush=True,
                )
                return False
            break
    except Exception:
        return True
    return True

def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, '').strip().lower()
    if not raw:
        return bool(default)
    return raw not in ('0', 'false', 'no', 'off')

def _pg_softplus(x: float) -> float:
    if x > 50.0:
        return float(x)
    if x < -50.0:
        return float(math.exp(x))
    return float(math.log1p(math.exp(x)))

def _cache_put(cache: dict, key, value, cap: int) -> None:
    if key not in cache and len(cache) >= cap:
        cache.pop(next(iter(cache)), None)
    cache[key] = value

def _safe_raise(round_state: RoundState, active: int, fraction: float) -> Union[RaiseAction, CheckAction, CallAction]:
    if RaiseAction not in round_state.legal_actions():
        continue_cost = round_state.pips[1 - active] - round_state.pips[active]
        return CallAction() if continue_cost > 0 and CallAction in round_state.legal_actions() else CheckAction()

    min_raise, max_raise = round_state.raise_bounds()
    amount = int(min_raise + (max_raise - min_raise) * max(0.0, min(1.0, float(fraction))))
    amount = max(min_raise, min(max_raise, amount))
    return RaiseAction(amount)
    
class GoodBot(Bot):

    RIVER_GATE_MARGIN = 0.06

    CHECK_RAISE_MIN_EQUITY = 0.72
    CHECK_RAISE_MIN_AGGR = 0.32
    CHECK_RAISE_FREQ = 0.45

    def __init__(self):
        self.kept_cards = None
        self.hand_strength_cache = {}
        self._equity_mc_cache = {}
        self._opp_sampler_cache = {}
        self._preflop_3card_table = {}
        self._river_equity_cache = {}
        self._river_exact_cache = {}
        self._blocker_cache = {}
        self._full_deck_codes = None
        self._opp_flop_checked = False
        self._our_raises = 0
        self._opp_folds_to_our_raises = 0
        self._last_action_was_raise = False
        self._opp_aggr_history = deque(maxlen=50)
        self._this_round_opp_aggr = 0
        self._this_round_opp_actions = 0
        self._last_seen_opp_pip = 0
        self._last_seen_street = None
        self._avg_opp_aggr = 0.0
        self._self_strategy_history = deque(maxlen=50)
        self._this_round_our_raises = 0
        self._this_round_our_calls = 0
        self._this_round_our_folds = 0
        self._this_round_our_checks = 0
        self._this_round_our_discards = 0
        self._this_round_our_bluffs = 0
        self._this_round_our_valuebets = 0
        self._precomputed_equity = None
        self._precomputed_bias_step = None
        self._precomputed_hand_buckets = None
        self._opp_stats = {
            'vpip': deque(maxlen=100),
            'pfr': deque(maxlen=100),
            'fold_to_raise': deque(maxlen=100)
        }
        self._observed_opp_hands = deque(maxlen=50)
        self._opp_river_bets_seen = 0
        self._opp_river_bluffs_seen = 0
        self._river_bluff_shrinkage = 10
        self._opp_bet_river_this_round = False
        self._mccfr_policy = None
        self._mccfr_score_ranges = {}
        self._mccfr_hand_buckets = HAND_BUCKETS
        self._mccfr_board_buckets = BOARD_BUCKETS
        self._mccfr_pot_buckets = POT_BB_BUCKETS
        self._mccfr_spr_buckets = SPR_BUCKETS
        self._mccfr_pressure_buckets = PRESSURE_BUCKETS
        self._mccfr_raise_fractions = MCCFR_RAISE_FRACTIONS
        self._use_mccfr = True
        self._mccfr_strict = _env_flag('GOODBOT_MCCFR_STRICT', False)
        self._mccfr_deterministic = _env_flag('GOODBOT_MCCFR_DETERMINISTIC', True)
        self._mccfr_fallback_iters = int(os.environ.get('GOODBOT_MCCFR_ITERS', '40') or 40)
        self._mccfr_discard_iters = int(os.environ.get('GOODBOT_MCCFR_DISCARD_ITERS', '50') or 50)
        self._mccfr_preflop_iters = int(os.environ.get('GOODBOT_MCCFR_PREFLOP_ITERS', '60') or 60)
        self._mccfr_map_river = _env_flag('GOODBOT_MCCFR_RIVER_MAP', True)
        self._mccfr_last_street = None
        self._mccfr_raise_count = 0
        self._mccfr_last_aggressor = -1
        self._mccfr_last_bet_bucket = 0
        self._mccfr_last_pips = [0, 0]
        self._mccfr_stats = {
            'total': 0,
            'hits': 0,
            'misses': 0,
            'fallback_hits': 0,
            'miss_by_street': {},
            'miss_reason': {},
        }
        self._mccfr_last_miss_reason = None
        self._mccfr_last_fallback_used = False
        self._mccfr_policy_traversals = None
        self._mccfr_reported = False
        if os.environ.get('GOODBOT_MCCFR_DISABLE', '').strip():
            self._use_mccfr = False
        self._use_river_logic = _env_flag('GOODBOT_RIVER_LOGIC', False)
        self._use_river_betting = _env_flag('GOODBOT_RIVER_BETTING', False)
        self._iter_scale = _parse_float_env('GOODBOT_ITER_SCALE', 1.0)
        if self._iter_scale <= 0.0:
            self._iter_scale = 1.0
        self._use_discard_read = _env_flag('GOODBOT_DISCARD_READ', False)
        self._use_range_track = _env_flag('GOODBOT_RANGE_TRACK', False)
        self._use_blockers = _env_flag('GOODBOT_BLOCKERS', False)
        self._use_check_raise = _env_flag('GOODBOT_CHECK_RAISE', False)
        self._use_lock_in = _env_flag('GOODBOT_LOCK_IN', True)
        self._pg = None
        self._use_pg = _env_flag('GOODBOT_USE_PG', False)
        if self._use_pg:
            self._pg = self._load_pg_policy()
            if self._pg is None:
                self._use_pg = False
       
        def _merge_table(path_to_load: str):
            try:
                opener = gzip.open if path_to_load.lower().endswith(('.gz', '.gzip')) else open
                with opener(path_to_load, 'rb') as f:
                    t = pickle.load(f)
                if isinstance(t, dict):
                    if self._precomputed_equity is None:
                        self._precomputed_equity = {}
                    self._precomputed_equity.update(t)
            except Exception:
                return
        path = os.environ.get('GOODBOT_PRECOMPUTED_EQUITY', '').strip()
        if path:
            _merge_table(path)
        else:
            try:
                base_dir = os.path.dirname(__file__)
                for name in (
                    'goodbot_precomputed_equity.pkl.gz',
                    'goodbot_precomputed_equity.pkl',
                    'goodbot_precomputed_equity_full.pkl.gz',
                    'goodbot_precomputed_equity_texture.pkl.gz',
                ):
                    candidate = os.path.join(base_dir, name)
                    if os.path.exists(candidate):
                        _merge_table(candidate)
            except Exception:
                pass
        if isinstance(self._precomputed_equity, dict) and self._precomputed_equity:
            try:
                bias_vals = set()
                max_hb = None
                for k in self._precomputed_equity.keys():
                    if isinstance(k, tuple) and len(k) == 3:
                        b = k[2]
                        if isinstance(b, (int, float)):
                            bias_vals.add(float(b))
                    if isinstance(k, tuple) and len(k) == 5 and k and k[0] == 'tex':
                        hb = k[1]
                        if isinstance(hb, int):
                            if max_hb is None or hb > max_hb:
                                max_hb = hb
                        b = k[4]
                        if isinstance(b, (int, float)):
                            bias_vals.add(float(b))
                    if len(bias_vals) >= 64 and max_hb is not None:
                        break
                step = None
                if len(bias_vals) >= 2:
                    svals = sorted(bias_vals)
                    diffs = [svals[i + 1] - svals[i] for i in range(len(svals) - 1)]
                    diffs = [d for d in diffs if d > 1e-9]
                    if diffs:
                        step = min(diffs)
                if step is not None and 0.0 < step <= 0.5:
                    self._precomputed_bias_step = round(step, 4)
                if max_hb is not None:
                    self._precomputed_hand_buckets = int(max_hb) + 1
            except Exception:
                pass
        try:
            policy_path = os.environ.get('GOODBOT_MCCFR_POLICY', '').strip()
            if not policy_path:
                for cand in (os.path.join(os.path.dirname(__file__), 'mccfr_policy.pkl'),
                             os.path.join(os.path.dirname(__file__), '..', 'tools', 'mccfr_policy.pkl')):
                    if os.path.exists(cand):
                        policy_path = cand
                        break
            if policy_path and os.path.exists(policy_path):
                with open(policy_path, 'rb') as f:
                    payload = pickle.load(f)
                self._mccfr_policy = payload.get('policy')
                if not _mccfr_policy_schema_ok(self._mccfr_policy, policy_path):
                    self._mccfr_policy = None
                self._mccfr_score_ranges = payload.get('score_ranges', {}) or {}
                self._mccfr_hand_buckets = payload.get('hand_buckets', self._mccfr_hand_buckets)
                self._mccfr_board_buckets = payload.get('board_buckets', self._mccfr_board_buckets)
                self._mccfr_pot_buckets = payload.get('pot_buckets', self._mccfr_pot_buckets)
                self._mccfr_spr_buckets = payload.get('spr_buckets', self._mccfr_spr_buckets)
                self._mccfr_pressure_buckets = payload.get('pressure_buckets', self._mccfr_pressure_buckets)
                self._mccfr_raise_fractions = payload.get('raise_fractions', self._mccfr_raise_fractions)
                self._mccfr_policy_traversals = payload.get('traversals')
        except Exception:
            self._mccfr_policy = None
        try:
            if self._mccfr_policy is None:
                ckpt_path = os.environ.get('GOODBOT_MCCFR_CHECKPOINT', '').strip()
                base_dir = os.path.dirname(__file__)
                candidates = [
                    ckpt_path,
                    os.path.join(base_dir, 'mccfr_ckpt.pkl'),
                    os.path.join(base_dir, '..', 'tools', 'mccfr_ckpt.pkl'),
                ]
                for path in candidates:
                    if path and os.path.exists(path):
                        with open(path, 'rb') as f:
                            payload = pickle.load(f)
                        strat_sum = payload.get('strategy_sum', {})
                        policy = {}
                        for info, sums in strat_sum.items():
                            try:
                                arr = np.array(sums, dtype=np.float32)
                                total = float(arr.sum())
                                if total > 1e-12:
                                    policy[info] = (arr / total).tolist()
                                else:
                                    policy[info] = arr.tolist()
                            except Exception:
                                continue
                        self._mccfr_policy = policy or None
                        if not _mccfr_policy_schema_ok(self._mccfr_policy, path):
                            self._mccfr_policy = None
                            break
                        self._mccfr_score_ranges = payload.get('score_ranges', {}) or {}
                        self._mccfr_hand_buckets = payload.get('hand_buckets', self._mccfr_hand_buckets)
                        self._mccfr_board_buckets = payload.get('board_buckets', self._mccfr_board_buckets)
                        self._mccfr_pot_buckets = payload.get('pot_buckets', self._mccfr_pot_buckets)
                        self._mccfr_spr_buckets = payload.get('spr_buckets', self._mccfr_spr_buckets)
                        self._mccfr_pressure_buckets = payload.get('pressure_buckets', self._mccfr_pressure_buckets)
                        self._mccfr_raise_fractions = payload.get('raise_fractions', self._mccfr_raise_fractions)
                        self._mccfr_policy_traversals = payload.get('traversals')
                        break
        except Exception:
            pass
   
    def _estimate_opp_tightness(self) -> float:
        vpip = 0.30
        pfr = 0.20
        fold_to_raise = 0.50
        avg_aggr = float(getattr(self, '_avg_opp_aggr', 0.0) or 0.0)
        try:
            if self._opp_stats['vpip']:
                vpip = float(sum(self._opp_stats['vpip'])) / float(len(self._opp_stats['vpip']))
            if self._opp_stats['pfr']:
                pfr = float(sum(self._opp_stats['pfr'])) / float(len(self._opp_stats['pfr']))
            if self._opp_stats['fold_to_raise']:
                fold_to_raise = float(sum(self._opp_stats['fold_to_raise'])) / float(len(self._opp_stats['fold_to_raise']))
        except Exception:
            pass
        tightness = 0.0
        tightness += (0.30 - vpip) / 0.20
        tightness += (0.22 - pfr) / 0.20
        tightness += 0.35 * (fold_to_raise - 0.50)
        tightness -= 0.8 * (avg_aggr - 0.25)
        return max(-1.0, min(1.0, float(tightness)))

    def _tightness_bucket(self, tightness: float) -> float:
        step = 0.25
        b = round(float(tightness) / step) * step
        return max(-1.0, min(1.0, b))

    def _three_card_class_key(self, cards) -> tuple:
        try:
            ranks = sorted([_RANK_TO_INT.get(_card_rank_char(c), 2) for c in cards], reverse=True)
            suits = [_card_suit_char(c) for c in cards]
        except Exception:
            ranks = [2, 2, 2]
            suits = ['c', 'd', 'h']
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        suited_class = tuple(sorted(suit_counts.values(), reverse=True))
        gaps = []
        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks)):
                gaps.append(abs(ranks[i] - ranks[j]))
        gaps.sort()
        return (tuple(ranks), suited_class, tuple(gaps))

    def _mccfr_effective_street(self, street: int) -> int:
        try:
            s = int(street)
        except Exception:
            s = 0
        return 5 if (s == 6 and getattr(self, '_mccfr_map_river', True)) else s

    def _mccfr_reset_history(self, round_state):
        self._mccfr_last_street = round_state.street
        self._mccfr_raise_count = 0
        self._mccfr_last_aggressor = -1
        self._mccfr_last_bet_bucket = 0
        self._mccfr_last_pips = list(round_state.pips)

    def _mccfr_bet_bucket_from_amount(self, round_state, amount, continue_cost=0, is_allin=False, pot_before=None):
        if is_allin:
            return 6
        eff_street = self._mccfr_effective_street(round_state.street)
        pot_after = STARTING_STACK - round_state.stacks[0] + STARTING_STACK - round_state.stacks[1]
        if pot_before is None:
            pot_before = pot_after
        try:
            pot_before = max(0.0, float(pot_before))
        except Exception:
            pot_before = float(pot_after)
        try:
            cont = max(0.0, float(continue_cost))
        except Exception:
            cont = 0.0
        base = max(1.0, pot_before + cont)
        try:
            ratio = (float(amount) - cont) / base
        except Exception:
            ratio = 0.0
        ratio = max(0.0, ratio)
        fractions = []
        try:
            frac_map = self._mccfr_raise_fractions.get(eff_street)
            if not isinstance(frac_map, dict):
                frac_map = self._mccfr_raise_fractions.get(str(eff_street))
            if not isinstance(frac_map, dict):
                frac_map = self._mccfr_raise_fractions.get('default', {})
            for idx in MCCFR_ALLOWED_RAISES:
                if idx == MCCFR_RAISE_ALLIN:
                    continue
                if isinstance(frac_map, dict) and idx in frac_map:
                    fractions.append((idx, float(frac_map[idx])))
        except Exception:
            fractions = []
        if not fractions:
            for idx in MCCFR_ALLOWED_RAISES:
                if idx == MCCFR_RAISE_ALLIN:
                    continue
                fractions.append((idx, get_raise_fraction(eff_street, idx)))
        best_idx = fractions[0][0]
        best_dist = abs(ratio - fractions[0][1])
        for idx, frac in fractions[1:]:
            dist = abs(ratio - frac)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return int(best_idx - MCCFR_RAISE_XSMALL + 1)

    def _mccfr_sync_history(self, round_state, active):
        if self._mccfr_last_street is None or self._mccfr_last_street != round_state.street:
            self._mccfr_reset_history(round_state)
            return
        opp = 1 - active
        try:
            last_opp = self._mccfr_last_pips[opp]
            last_me = self._mccfr_last_pips[active]
            if round_state.pips[opp] > last_opp:
                if round_state.pips[opp] > last_me:
                    self._mccfr_raise_count = min(3, self._mccfr_raise_count + 1)
                    self._mccfr_last_aggressor = opp
                    bet_amount = round_state.pips[opp] - last_opp
                    continue_cost = max(0, last_me - last_opp)
                    is_allin = round_state.stacks[opp] == 0
                    pot_after = STARTING_STACK - round_state.stacks[0] + STARTING_STACK - round_state.stacks[1]
                    pot_before = pot_after - bet_amount
                    self._mccfr_last_bet_bucket = self._mccfr_bet_bucket_from_amount(
                        round_state,
                        bet_amount,
                        continue_cost=continue_cost,
                        is_allin=is_allin,
                        pot_before=pot_before,
                    )
            self._mccfr_last_pips = list(round_state.pips)
        except Exception:
            self._mccfr_last_pips = list(round_state.pips)

    def _mccfr_note_action(self, round_state, active, action, action_idx=None):
        try:
            if isinstance(action, RaiseAction):
                self._mccfr_raise_count = min(3, self._mccfr_raise_count + 1)
                self._mccfr_last_aggressor = active
                min_raise, max_raise = round_state.raise_bounds()
                bet_amount = action.amount - round_state.pips[active]
                is_allin = action.amount >= max_raise
                if action_idx is not None:
                    if action_idx == MCCFR_RAISE_ALLIN or is_allin:
                        self._mccfr_last_bet_bucket = 6
                    else:
                        self._mccfr_last_bet_bucket = int(action_idx - MCCFR_RAISE_XSMALL + 1)
                else:
                    continue_cost = round_state.pips[1 - active] - round_state.pips[active]
                    self._mccfr_last_bet_bucket = self._mccfr_bet_bucket_from_amount(
                        round_state,
                        bet_amount,
                        continue_cost=continue_cost,
                        is_allin=is_allin,
                    )
                self._mccfr_last_pips[active] = action.amount
            elif isinstance(action, CallAction):
                cont = round_state.pips[1 - active] - round_state.pips[active]
                if cont > 0:
                    self._mccfr_last_pips[active] = round_state.pips[active] + cont
            elif isinstance(action, CheckAction):
                pass
        except Exception:
            pass

    def _mccfr_info_key(self, round_state, active):
        eff_street = self._mccfr_effective_street(round_state.street)
        pot = STARTING_STACK - round_state.stacks[0] + STARTING_STACK - round_state.stacks[1]
        pot_bb = float(pot) / float(BIG_BLIND)
        continue_cost = round_state.pips[1 - active] - round_state.pips[active]
        pressure = float(continue_cost) / float(max(1.0, pot + continue_cost))
        facing_raise = bool(continue_cost > 0 and round_state.pips[active] > 0)
        raise_count_bucket = min(2, int(self._mccfr_raise_count))
        if self._mccfr_last_aggressor < 0:
            last_aggressor = 0
        else:
            last_aggressor = 1 if self._mccfr_last_aggressor == active else 2
        last_bet_bucket = int(self._mccfr_last_bet_bucket)
        default_hand_buckets = select_bucket_count(HAND_BUCKETS, eff_street, 8)
        default_board_buckets = select_bucket_count(BOARD_BUCKETS, eff_street, 32)
        hand_bucket_count = select_bucket_count(self._mccfr_hand_buckets, eff_street, default_hand_buckets)
        board_bucket_count = select_bucket_count(self._mccfr_board_buckets, eff_street, default_board_buckets)
        hand_bucket = hand_strength_bucket(
            round_state.hands[active],
            round_state.board,
            self._mccfr_score_ranges,
            bucket_count=hand_bucket_count,
        )
        board_bucket = board_texture_bucket(round_state.board, bucket_count=board_bucket_count)
        shape = three_card_shape(round_state.hands[active])
        return mccfr_info_key(
            eff_street,
            active,
            pot_bb,
            float(round_state.stacks[active]) / float(max(1.0, pot)),
            pressure,
            facing_raise,
            raise_count_bucket,
            last_aggressor,
            last_bet_bucket,
            hand_bucket,
            board_bucket,
            shape,
            pot_buckets=self._mccfr_pot_buckets,
            spr_buckets=self._mccfr_spr_buckets,
            pressure_buckets=self._mccfr_pressure_buckets,
        )

    def _mccfr_legal_action_mask(self, round_state, active):
        legal_actions = round_state.legal_actions()
        mask = [0.0] * len(MCCFR_ACTIONS)
        continue_cost = round_state.pips[1 - active] - round_state.pips[active]
        if DiscardAction in legal_actions:
            hand = round_state.hands[active]
            if len(hand) >= 1:
                mask[MCCFR_DISCARD_0] = 1.0
            if len(hand) >= 2:
                mask[MCCFR_DISCARD_1] = 1.0
            if len(hand) >= 3:
                mask[MCCFR_DISCARD_2] = 1.0
            return mask
        if CallAction in legal_actions or CheckAction in legal_actions:
            mask[MCCFR_CALL_CHECK] = 1.0
        if FoldAction in legal_actions and continue_cost > 0:
            mask[MCCFR_FOLD] = 1.0
        if RaiseAction in legal_actions:
            for i in MCCFR_ALLOWED_RAISES:
                mask[i] = 1.0
        return mask

    def _mccfr_raise_amount(self, round_state, active, action_idx):
        min_raise, max_raise = round_state.raise_bounds()
        if action_idx == MCCFR_RAISE_ALLIN:
            return max_raise
        frac = None
        eff_street = self._mccfr_effective_street(round_state.street)
        try:
            fractions = self._mccfr_raise_fractions
            if isinstance(fractions, dict):
                frac_map = fractions.get(eff_street)
                if not isinstance(frac_map, dict):
                    frac_map = fractions.get(str(eff_street))
                if not isinstance(frac_map, dict):
                    frac_map = fractions.get('default', {})
                if isinstance(frac_map, dict) and action_idx in frac_map:
                    frac = float(frac_map[action_idx])
        except Exception:
            frac = None
        if frac is None:
            frac = get_raise_fraction(eff_street, action_idx)
        pot = STARTING_STACK - round_state.stacks[0] + STARTING_STACK - round_state.stacks[1]
        continue_cost = round_state.pips[1 - active] - round_state.pips[active]
        pot_after_call = pot + max(0, continue_cost)
        base = max(1.0, float(pot_after_call))
        amount = int(round_state.pips[active] + max(0, continue_cost) + base * float(frac))
        return max(min_raise, min(max_raise, amount))

    def _mccfr_action_from_index(self, round_state, active, action_idx):
        legal_actions = round_state.legal_actions()
        if action_idx == MCCFR_FOLD and FoldAction in legal_actions:
            action = FoldAction()
            return action
        if action_idx == MCCFR_CALL_CHECK and (CallAction in legal_actions or CheckAction in legal_actions):
            continue_cost = round_state.pips[1 - active] - round_state.pips[active]
            if continue_cost > 0 and CallAction in legal_actions:
                action = CallAction()
                self._mccfr_note_action(round_state, active, action, action_idx)
                return action
            if CheckAction in legal_actions:
                action = CheckAction()
                self._mccfr_note_action(round_state, active, action, action_idx)
                return action
        if action_idx in (MCCFR_RAISE_XSMALL, MCCFR_RAISE_SMALL, MCCFR_RAISE_MED, MCCFR_RAISE_LARGE, MCCFR_RAISE_OVER, MCCFR_RAISE_ALLIN) and RaiseAction in legal_actions:
            amount = self._mccfr_raise_amount(round_state, active, action_idx)
            action = RaiseAction(amount)
            self._mccfr_note_action(round_state, active, action, action_idx)
            return action
        if action_idx in (MCCFR_DISCARD_0, MCCFR_DISCARD_1, MCCFR_DISCARD_2) and DiscardAction in legal_actions:
            discard_idx = action_idx - MCCFR_DISCARD_0
            hand = round_state.hands[active]
            if 0 <= discard_idx < len(hand):
                action = DiscardAction(discard_idx)
                return action
        return None

    def _mccfr_fallback_probs(self, info):
        if not self._mccfr_policy:
            return None
        try:
            base = list(info)
        except Exception:
            return None

        def _coarsen(value, factor=2):
            try:
                v = int(value)
            except Exception:
                return 0
            if factor <= 1:
                return v
            return max(0, v // factor)

        candidates = []
        c = list(base)
        c[9] = 0
        candidates.append(('drop_last_bet', tuple(c)))
        c = list(base)
        c[7] = 0
        c[8] = 0
        c[9] = 0
        candidates.append(('drop_aggr', tuple(c)))
        c = list(base)
        c[10] = _coarsen(c[10])
        c[11] = _coarsen(c[11])
        candidates.append(('coarsen_hand_board', tuple(c)))
        c = list(base)
        c[3] = _coarsen(c[3])
        c[4] = _coarsen(c[4])
        c[5] = _coarsen(c[5])
        candidates.append(('coarsen_pot_spr_pressure', tuple(c)))
        c = list(base)
        c[12] = 0
        c[13] = 0
        c[14] = 0
        candidates.append(('drop_shape', tuple(c)))
        c = list(base)
        c[7] = 0
        c[8] = 0
        c[9] = 0
        c[10] = _coarsen(c[10])
        c[11] = _coarsen(c[11])
        c[12] = 0
        c[13] = 0
        c[14] = 0
        candidates.append(('coarsen_combo', tuple(c)))

        for name, key in candidates:
            probs = self._mccfr_policy.get(key)
            if probs:
                self._mccfr_last_miss_reason = f'fallback:{name}'
                return probs
        return None

    def _mccfr_select_action(self, round_state, active):
        if not self._mccfr_policy:
            self._mccfr_last_miss_reason = 'no_policy'
            return None
        info = self._mccfr_info_key(round_state, active)
        self._mccfr_last_fallback_used = False
        probs = self._mccfr_policy.get(info)
        if not probs:
            probs = self._mccfr_fallback_probs(info)
            if probs:
                self._mccfr_last_fallback_used = True
            else:
                self._mccfr_last_miss_reason = 'no_probs'
                return None
        mask = self._mccfr_legal_action_mask(round_state, active)
        weights = [float(p) * float(m) for p, m in zip(probs, mask)]
        total = sum(weights)
        if total <= 1e-12:
            self._mccfr_last_miss_reason = 'zero_weights'
            return None
        self._mccfr_last_miss_reason = None
        if self._mccfr_deterministic:
            idx = max(range(len(weights)), key=lambda i: weights[i])
            return self._mccfr_action_from_index(round_state, active, idx)
        r = random.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                return self._mccfr_action_from_index(round_state, active, i)
        return self._mccfr_action_from_index(round_state, active, len(weights) - 1)
   
    def handle_new_round(self, game_state, round_state, active):
        self.kept_cards = None
        self._last_action_was_raise = False
        try:
            self._this_round_opp_aggr = 0
            self._this_round_opp_actions = 0
            self._opp_aggr_streets = set()
            self._opp_max_pressure = 0.0
            self._last_seen_opp_pip = 0
            self._last_seen_street = None
            self._this_round_our_raises = 0
            self._this_round_our_calls = 0
            self._this_round_our_folds = 0
            self._this_round_our_checks = 0
            self._this_round_our_discards = 0
            self._this_round_our_bluffs = 0
            self._this_round_our_valuebets = 0
            self._opp_flop_checked = False
            self._opp_bet_river_this_round = False
            self._opp_vpip_this_round = False
            self._opp_pfr_this_round = False
            self._mccfr_last_street = round_state.street
            self._mccfr_raise_count = 0
            self._mccfr_last_aggressor = -1
            self._mccfr_last_bet_bucket = 0
            self._mccfr_last_pips = list(round_state.pips)
        except Exception:
            pass
   
    def handle_round_over(self, game_state, terminal_state, active):
        try:
            prev = terminal_state.previous_state
            opp_cards_raw = prev.hands[1 - active]
            my_delta = terminal_state.deltas[active]
            if self._last_action_was_raise and my_delta > 0 and (not opp_cards_raw):
                self._opp_folds_to_our_raises += 1
                self._opp_stats['fold_to_raise'].append(1.0)
            elif self._last_action_was_raise:
                self._opp_stats['fold_to_raise'].append(0.0)
        except Exception:
            pass

        try:
            self._opp_stats['vpip'].append(1.0 if getattr(self, '_opp_vpip_this_round', False) else 0.0)
            self._opp_stats['pfr'].append(1.0 if getattr(self, '_opp_pfr_this_round', False) else 0.0)
        except Exception:
            pass
   
        try:
            if opp_cards_raw and len(opp_cards_raw) == 2:
                opp_hand = tuple(sorted(str(c) for c in opp_cards_raw))
                board_key = tuple(sorted(str(c) for c in (prev.board or [])))
                opp_delta = terminal_state.deltas[1 - active]
                action_type = 'value' if opp_delta > 0 else 'bluff'
                self._observed_opp_hands.append((opp_hand, board_key, action_type))
                if self._opp_bet_river_this_round and len(prev.board or []) == 6:
                    pct = self._river_equity_vs_top_quantile(
                        list(opp_cards_raw), list(prev.board), 0.0)
                    self._opp_river_bets_seen += 1
                    if pct < 0.55:
                        self._opp_river_bluffs_seen += 1
        except Exception:
            pass
   
        try:
            aggr = 0.0
            if self._this_round_opp_actions > 0:
                aggr = float(self._this_round_opp_aggr) / float(self._this_round_opp_actions)
            self._opp_aggr_history.append(aggr)
            self._avg_opp_aggr = float(sum(self._opp_aggr_history)) / max(1, len(self._opp_aggr_history))
        except Exception:
            pass
   
        try:
            my_delta = float(terminal_state.deltas[active]) if hasattr(terminal_state, 'deltas') else 0.0
            raises = getattr(self, '_this_round_our_raises', 0)
            calls = getattr(self, '_this_round_our_calls', 0)
            folds = getattr(self, '_this_round_our_folds', 0)
            checks = getattr(self, '_this_round_our_checks', 0)
            discards = getattr(self, '_this_round_our_discards', 0)
            bluffs = getattr(self, '_this_round_our_bluffs', 0)
            valuebets = getattr(self, '_this_round_our_valuebets', 0)
            total_actions = max(1, raises + calls + folds + checks + discards)
            raise_rate = float(raises) / float(total_actions)
            if bluffs > valuebets and raises > 0:
                strat = 'bluff_heavy'
            elif raise_rate >= 0.35:
                strat = 'aggressive'
            elif raises == 0 and calls > 0:
                strat = 'passive'
            else:
                strat = 'balanced'
            self._self_strategy_history.append({'strategy': strat, 'delta': my_delta})
        except Exception:
            pass

        try:
            if (not getattr(self, '_mccfr_reported', False)) and game_state.round_num >= NUM_ROUNDS:
                stats = getattr(self, '_mccfr_stats', {}) or {}
                total = int(stats.get('total', 0) or 0)
                hits = int(stats.get('hits', 0) or 0)
                misses = int(stats.get('misses', 0) or 0)
                fallback_hits = int(stats.get('fallback_hits', 0) or 0)
                miss_reason = stats.get('miss_reason', {}) or {}
                miss_by_street = stats.get('miss_by_street', {}) or {}
                policy_infosets = len(self._mccfr_policy) if isinstance(self._mccfr_policy, dict) else 0
                traversals = getattr(self, '_mccfr_policy_traversals', None)
                print(f"[mccfr] policy_infosets={policy_infosets} policy_traversals={traversals}")
                print(f"[mccfr] decisions={total} hits={hits} misses={misses} fallback_hits={fallback_hits}")
                if miss_reason:
                    print(f"[mccfr] miss_reasons={miss_reason}")
                if miss_by_street:
                    print(f"[mccfr] miss_by_street={miss_by_street}")
                self._mccfr_reported = True
        except Exception:
            pass
    
        self._last_action_was_raise = False
       
    def calculate_hand_potential(self, cards, board_cards, iters=200, opp_bias=0.0):
        try:
            cards_key = tuple(sorted(str(c) for c in cards))
            board_key = tuple(sorted(str(c) for c in (board_cards or [])))
            bias = max(0.0, min(1.0, float(opp_bias)))
            step = self._precomputed_bias_step or 0.1
            bias_bucket = round(bias / step) * step
            bias_bucket = round(max(0.0, min(1.0, bias_bucket)), 4)
            tightness = self._estimate_opp_tightness()
            tight_bucket = self._tightness_bucket(tightness)
            cache_key = ('potential', cards_key, board_key, int(iters))
            if cache_key in self.hand_strength_cache:
                return self.hand_strength_cache[cache_key]
        except Exception:
            cache_key = None
            bias_bucket = 0.0
            tight_bucket = 0.0
        if len(cards) == 3:
            try:
                class_key = self._three_card_class_key(cards)
                cache_key = ('potential3', class_key, int(iters), bias_bucket, tight_bucket)
                if cache_key in self._preflop_3card_table:
                    return self._preflop_3card_table[cache_key]
            except Exception:
                pass
            iters_total = max(40, int(iters))
            iters_per = max(20, iters_total // 2)
            equity_by_discard = []
            for discard_idx in (0, 1, 2):
                kept = [cards[i] for i in (0, 1, 2) if i != discard_idx]
                equity = self.calculate_equity(kept, [], iters=iters_per, opp_bias=opp_bias)
                equity_by_discard.append((equity, discard_idx))
            equity_by_discard.sort(reverse=True)
            if len(equity_by_discard) >= 2:
                gap = equity_by_discard[0][0] - equity_by_discard[1][0]
                if gap < 0.03 and iters_total >= 100:
                    top = equity_by_discard[:2]
                    equity_by_discard = []
                    for _, discard_idx in top:
                        kept = [cards[i] for i in (0, 1, 2) if i != discard_idx]
                        equity = self.calculate_equity(kept, [], iters=iters_total, opp_bias=opp_bias)
                        equity_by_discard.append((equity, discard_idx))
                    equity_by_discard.sort(reverse=True)
            best = equity_by_discard[0][0] if equity_by_discard else 0.0
            if cache_key is not None:
                self._preflop_3card_table[cache_key] = best
            return best
        else:
            ranks = [_RANK_TO_INT[_card_rank_char(c)] for c in cards]
            suited = bool(_card_suit_char(cards[0]) and _card_suit_char(cards[0]) == _card_suit_char(cards[1]))
           
            if ranks[0] == ranks[1]:
                result = 0.25 + ranks[0] / 20.0
            else:
                high_cards = sum(1 for r in ranks if r >= 11)
                result = 0.2 + 0.15 * high_cards + (0.1 if suited else 0.0)
            if cache_key is not None:
                _cache_put(self.hand_strength_cache, cache_key, result, 65536)
            return result
   
    def calculate_equity(self, my_cards, board_cards, iters=200, opp_bias=0.0):
        try:
            my_key = tuple(sorted(str(c) for c in my_cards))
            board_key = tuple(sorted(str(c) for c in (board_cards or [])))
            bias = max(0.0, min(1.0, float(opp_bias)))
            tightness = self._estimate_opp_tightness()
            tight_bucket = self._tightness_bucket(tightness)
            step = self._precomputed_bias_step or 0.1
            bias_bucket = round(bias / step) * step
            bias_bucket = round(max(0.0, min(1.0, bias_bucket)), 4)
            state_key = (my_key, board_key, bias_bucket)
        except Exception:
            state_key = None
            tightness = 0.0
            tight_bucket = 0.0
        mc_key = None
        if state_key is not None:
            mc_key = (state_key, tight_bucket)
        if state_key is not None and self._precomputed_equity:
            try:
                v = self._precomputed_equity.get(state_key)
                if v is None:
                    v = self._precomputed_equity.get((state_key[0], state_key[1]))
                if v is not None:
                    return float(v)
            except Exception:
                pass
            try:
                hbuckets = self._precomputed_hand_buckets or 20
                if len(my_key) == 2:
                    try:
                        c1, c2 = my_key[0], my_key[1]
                        r1 = _RANK_TO_INT.get(_card_rank_char(c1), 2)
                        r2 = _RANK_TO_INT.get(_card_rank_char(c2), 2)
                        hi = max(r1, r2)
                        lo = min(r1, r2)
                        pair = (r1 == r2)
                        suited = _are_suited(c1, c2)
                        gap = hi - lo
                        base_map = {14: 10.0, 13: 8.0, 12: 7.0, 11: 6.0, 10: 5.0, 9: 4.5, 8: 4.0, 7: 3.5, 6: 3.0, 5: 2.5, 4: 2.0, 3: 1.5, 2: 1.0}
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
                        strength = 1.0 / (1.0 + math.exp(-x))
                    except Exception:
                        strength = 0.5
                    hb = int(max(0, min(hbuckets - 1, int(float(strength) * hbuckets))))
                    tex = _board_texture_key(board_cards)
                    tex_key = ('tex', hb, len(board_key), tex, bias_bucket)
                    v2 = self._precomputed_equity.get(tex_key)
                    if v2 is None:
                        tex_old = _board_texture_key_legacy(board_cards)
                        tex_key_old = ('tex', hb, len(board_key), tex_old, bias_bucket)
                        v2 = self._precomputed_equity.get(tex_key_old)
                    if v2 is not None:
                        return float(v2)
            except Exception:
                pass
        remaining = 6 - len(board_cards)
        _card_cache = {}
        def _c(code: str):
            obj = _card_cache.get(code)
            if obj is None:
                obj = pkrbot.Card(code)
                _card_cache[code] = obj
            return obj
        my_codes = [str(card) for card in my_cards]
        board_codes = [str(card) for card in board_cards]
        my_hand = [_c(code) for code in my_codes]
        board_hand = [_c(code) for code in board_codes]
        def _safe_remove(deck, card) -> bool:
            try:
                if card in deck.cards:
                    deck.cards.remove(card)
                    return True
                return False
            except Exception:
                return False
        target_iters = int(iters)
        if target_iters <= 0:
            return 0.5
        bias = 0.0
        try:
            bias = max(0.0, min(1.0, float(opp_bias)))
        except Exception:
            bias = 0.0
        def _hand_strength_proxy(c1: str, c2: str) -> float:
            if self._precomputed_equity is not None:
                try:
                    key = (tuple(sorted((str(c1), str(c2)))), tuple(), 0.0)
                    v = self._precomputed_equity.get(key)
                    if v is None:
                        v = self._precomputed_equity.get((key[0], key[1]))
                    if v is not None:
                        return max(0.0, min(1.0, float(v)))
                except Exception:
                    pass
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
        def _get_opp_sampler():
            if mc_key is None:
                return None
            if mc_key in self._opp_sampler_cache:
                return self._opp_sampler_cache[mc_key]
            deck = pkrbot.Deck()
            for card in my_hand + board_hand:
                _safe_remove(deck, card)
            remaining_cards = [str(c) for c in deck.cards]
            combos = []
            cum = []
            total = 0.0
            n = len(remaining_cards)
            tight_k = 1.0 + 1.3 * float(tightness)
            tight_k = max(0.5, min(2.5, tight_k))
            sim_half = None
            try:
                recent_obs = list(self._observed_opp_hands)[-10:]
                if recent_obs:
                    inv = 1.0 / float(len(recent_obs))
                    valid = [obs[0] for obs in recent_obs if len(obs[0]) == 2]
                    sim_half = {}
                    for c in remaining_cards:
                        rc = _card_rank_char(c)
                        sc = _card_suit_char(c)
                        acc = 0.0
                        for o1, o2 in valid:
                            if _card_rank_char(o1) == rc:
                                acc += 0.5
                            if _card_suit_char(o1) == sc:
                                acc += 0.5
                            if _card_rank_char(o2) == rc:
                                acc += 0.5
                            if _card_suit_char(o2) == sc:
                                acc += 0.5
                        sim_half[c] = acc * inv
            except Exception:
                sim_half = None
            for i in range(n):
                c1 = remaining_cards[i]
                for j in range(i + 1, n):
                    c2 = remaining_cards[j]
                    strength = _hand_strength_proxy(c1, c2)
                    strength_w = max(1e-3, strength) ** tight_k
                    w = strength_w * math.exp(2.0 * bias * (strength - 0.5))
                    if sim_half is not None:
                        w *= 1.0 + 0.5 * (sim_half[c1] + sim_half[c2])
                    total += w
                    combos.append((c1, c2))
                    cum.append(total)
            sampler = (combos, cum, total, tuple(remaining_cards))
            _cache_put(self._opp_sampler_cache, mc_key, sampler, 512)
            return sampler
        opp_sampler = _get_opp_sampler()
        def _sample_opp_hand_strings():
            if not opp_sampler:
                deck = pkrbot.Deck()
                for card in my_hand + board_hand:
                    _safe_remove(deck, card)
                deck.shuffle()
                h = deck.deal(2)
                return (str(h[0]), str(h[1]))
            combos, cum, total, _remaining_cards = opp_sampler
            if not combos or total <= 0.0:
                deck = pkrbot.Deck()
                for card in my_hand + board_hand:
                    _safe_remove(deck, card)
                deck.shuffle()
                h = deck.deal(2)
                return (str(h[0]), str(h[1]))
            r = random.random() * total
            idx = bisect.bisect_left(cum, r)
            if idx >= len(combos):
                idx = len(combos) - 1
            return combos[idx]
        def _one_rollout() -> float:
            if opp_sampler:
                base_remaining = opp_sampler[3]
            else:
                deck = pkrbot.Deck()
                for code in my_codes + board_codes:
                    try:
                        deck.cards.remove(_c(code))
                    except Exception:
                        pass
                base_remaining = tuple(str(c) for c in deck.cards)
            o1, o2 = _sample_opp_hand_strings()
            if o1 == o2 or o1 not in base_remaining or o2 not in base_remaining:
                pool = list(base_remaining)
                if len(pool) >= 2:
                    o1, o2 = random.sample(pool, 2)
            if remaining > 0:
                pool = [c for c in base_remaining if c != o1 and c != o2]
                remaining_board = [_c(code) for code in random.sample(pool, remaining)]
            else:
                remaining_board = []
            opp_hand = [_c(o1), _c(o2)]
            full_board = board_hand + remaining_board
            my_strength = pkrbot.evaluate(my_hand + full_board)
            opp_strength = pkrbot.evaluate(opp_hand + full_board)
            if my_strength > opp_strength:
                return 1.0
            if my_strength == opp_strength:
                return 0.5
            return 0.0
        if remaining == 0 and opp_sampler:
            if mc_key is not None and mc_key in self._river_exact_cache:
                return float(self._river_exact_cache[mc_key])
            try:
                _combos, _cum, _total_w, _ = opp_sampler
                if _combos and _total_w > 0.0:
                    my_strength = pkrbot.evaluate(my_hand + board_hand)
                    acc = 0.0
                    prev = 0.0
                    for _i, (_o1, _o2) in enumerate(_combos):
                        w = _cum[_i] - prev
                        prev = _cum[_i]
                        if w <= 0.0:
                            continue
                        opp_strength = pkrbot.evaluate([_c(_o1), _c(_o2)] + board_hand)
                        if my_strength > opp_strength:
                            acc += w
                        elif my_strength == opp_strength:
                            acc += 0.5 * w
                    val = float(acc / _total_w)
                    if mc_key is not None:
                        self._river_exact_cache[mc_key] = val
                    return val
            except Exception:
                pass
        if mc_key is not None and mc_key in self._equity_mc_cache:
            entry = self._equity_mc_cache[mc_key]
        else:
            entry = {'samples': [], 'sum': 0.0}
            if mc_key is not None:
                _cache_put(self._equity_mc_cache, mc_key, entry, 4096)
        samples = entry.get('samples')
        if not isinstance(samples, list):
            samples = []
            entry['samples'] = samples
        total_sum = float(entry.get('sum', 0.0) or 0.0)
        if len(samples) < target_iters:
            for _ in range(target_iters - len(samples)):
                v = _one_rollout()
                samples.append(v)
                total_sum += v
        refresh = 0
        for _ in range(refresh):
            idx = random.randrange(target_iters)
            old = samples[idx]
            v = _one_rollout()
            samples[idx] = v
            total_sum += (v - old)
        if len(samples) != target_iters:
            total_sum = float(sum(samples))
        entry['sum'] = total_sum
        return total_sum / max(1, len(samples))
    def _get_full_deck_codes(self):
        if self._full_deck_codes is not None:
            return self._full_deck_codes
        ranks = '23456789TJQKA'
        suits = 'cdhs'
        self._full_deck_codes = [r + s for r in ranks for s in suits]
        return self._full_deck_codes
    def _river_equity_vs_top_quantile(self, my2, board6, quantile: float) -> float:
        try:
            q = max(0.0, min(0.99, float(quantile)))
        except Exception:
            q = 0.9
        try:
            my_key = tuple(sorted(str(c) for c in my2))
            board_key = tuple(str(c) for c in board6)
            cache_key = ('river_q', my_key, board_key, round(q, 3))
        except Exception:
            cache_key = None
        if cache_key is not None and cache_key in self._river_equity_cache:
            return float(self._river_equity_cache[cache_key])
        my_codes = [str(c) for c in my2]
        board_codes = [str(c) for c in board6]
        if len(my_codes) != 2 or len(board_codes) != 6:
            return 0.5
        dead = set(my_codes) | set(board_codes)
        remaining = [c for c in self._get_full_deck_codes() if c not in dead]
        if len(remaining) < 2:
            return 0.5
        board_hand = [pkrbot.Card(c) for c in board_codes]
        my_hand = [pkrbot.Card(c) for c in my_codes]
        my_score = int(pkrbot.evaluate(board_hand + my_hand))
        opp_scores = []
        opp_score_counts = {}
        for o1, o2 in combinations(remaining, 2):
            score = int(pkrbot.evaluate(board_hand + [pkrbot.Card(o1), pkrbot.Card(o2)]))
            opp_scores.append(score)
            opp_score_counts[score] = opp_score_counts.get(score, 0) + 1
        n = len(opp_scores)
        if n <= 0:
            return 0.5
        sorted_scores = sorted(opp_scores)
        cutoff = sorted_scores[int(round(q * (n - 1)))]
        wins = 0
        ties = 0
        total = 0
        for score in opp_scores:
            if score < cutoff:
                continue
            total += 1
            if my_score > score:
                wins += 1
            elif my_score == score:
                ties += 1
        equity = 0.5 if total <= 0 else (wins + 0.5 * ties) / total
        if cache_key is not None:
            self._river_equity_cache[cache_key] = float(equity)
        return float(equity)
    def _river_quantile_from_pressure(self, pressure: float, my_pip: int) -> float:
        p = max(0.0, min(1.0, float(pressure)))
        if p >= 0.55:
            q = 0.75
        elif p >= 0.40:
            q = 0.70
        elif p >= 0.25:
            q = 0.62
        else:
            q = 0.50
        if my_pip > 0:
            q = min(0.85, q + 0.05)
        return q

    def _opp_discard_card(self, round_state, active):
        try:
            board = list(getattr(round_state, 'board', []) or [])
            if len(board) < 4:
                return None
            return board[2] if int(active) == 0 else board[3]
        except Exception:
            return None

    def _opp_discard_bias(self, round_state, active) -> float:
        card = self._opp_discard_card(round_state, active)
        if card is None:
            return 0.0
        try:
            r = _RANK_TO_INT.get(_card_rank_char(card), 0)
        except Exception:
            return 0.0
        if r >= 11:
            return 0.10
        if r <= 6:
            return -0.05
        return 0.0

    def _blocker_strength(self, my2, board, quantile: float = 0.80) -> float:
        try:
            my_codes = [str(c) for c in my2]
            board_codes = [str(c) for c in board]
            if len(my_codes) != 2 or len(board_codes) < 4:
                return 1.0
            key = ('blk', tuple(sorted(my_codes)), tuple(board_codes), round(float(quantile), 3))
            if key in self._blocker_cache:
                return float(self._blocker_cache[key])
            avail = [c for c in self._get_full_deck_codes() if c not in set(board_codes)]
            bh = [pkrbot.Card(c) for c in board_codes]
            scored = []
            for a, b in combinations(avail, 2):
                scored.append((int(pkrbot.evaluate(bh + [pkrbot.Card(a), pkrbot.Card(b)])), a, b))
            if not scored:
                return 1.0
            scored.sort(key=lambda t: t[0])
            cut = scored[int(round(max(0.0, min(0.99, quantile)) * (len(scored) - 1)))][0]
            value = [(a, b) for s, a, b in scored if s >= cut]
            if not value:
                return 1.0
            mine = set(my_codes)
            blocked = sum(1 for a, b in value if a in mine or b in mine)
            n = len(avail)
            expected = 1.0 - ((n - 2) * (n - 3)) / float(n * (n - 1))
            ratio = (blocked / float(len(value))) / max(1e-9, expected)
            ratio = max(0.3, min(3.0, ratio))
            self._blocker_cache[key] = ratio
            return ratio
        except Exception:
            return 1.0

    def _river_bluff_prior(self, pressure: float) -> float:
        p = max(0.0, min(0.95, float(pressure)))
        s = p / max(1e-6, (1.0 - p))
        return max(0.0, min(0.5, s / (1.0 + 2.0 * s)))

    def _river_max_bluff(self, my_pip: int, pressure: float, facing_raise: bool,
                         board_completes_draw: bool, paired_board: bool) -> float:
        prior = self._river_bluff_prior(pressure)
        base = prior
        try:
            n = int(getattr(self, '_opp_river_bets_seen', 0) or 0)
            k = int(getattr(self, '_river_bluff_shrinkage', 10) or 10)
            if n > 0:
                obs = float(self._opp_river_bluffs_seen) / float(n)
                base = (n * obs + k * prior) / float(n + k)
        except Exception:
            base = prior
        try:
            if len(getattr(self, '_opp_aggr_history', ())) > 0:
                aggr = float(getattr(self, '_avg_opp_aggr', 0.25) or 0.0)
                base += 0.20 * (aggr - 0.25)
        except Exception:
            pass
        if facing_raise:
            base *= 0.70
        if board_completes_draw:
            base *= 0.90
        if paired_board:
            base *= 0.95
        return max(0.02, min(0.80, base))
    def _min_bluff_freq_for_call(self, required_equity: float, equity_vs_value: float) -> float:
        req = max(0.0, min(1.0, float(required_equity)))
        pv = max(0.0, min(1.0, float(equity_vs_value)))
        if pv >= 1.0:
            return 0.0
        f = (req - pv) / max(1e-9, (1.0 - pv))
        return max(0.0, min(1.0, f))
       
    def _board_completes_draw(self, board):
        tex = _board_texture_key(board)
        prev_tex = _board_texture_key(board[:-1])
        max_suit = tex[6] if len(tex) > 6 else 0
        best_run = tex[9] if len(tex) > 9 else 0
        prev_max_suit = prev_tex[6] if len(prev_tex) > 6 else 0
        prev_best_run = prev_tex[9] if len(prev_tex) > 9 else 0
        return (max_suit >= 4 and prev_max_suit < 4) or (best_run >= 5 and prev_best_run < 5)
    def _load_pg_policy(self):
        try:
            import pg_features
        except Exception:
            return None
        path = os.environ.get('GOODBOT_PG_WEIGHTS', '').strip()
        if not path:
            path = os.path.join(os.path.dirname(__file__), 'goodbot_pg_policy.json')
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r') as f:
                payload = json.load(f)
            if str(payload.get('model')) != 'mlp':
                return None
            feature_dim = int(payload.get('feature_dim') or 0)
            if feature_dim <= 0 or feature_dim > int(pg_features.feature_size()):
                _have = int(pg_features.feature_size())
                if feature_dim <= 0:
                    _why = "declares no usable feature_dim (got {!r})".format(
                        payload.get('feature_dim'))
                else:
                    _why = ("wants feature_dim={}, more than pg_features can "
                            "produce ({})".format(feature_dim, _have))
                print("[pg] policy DISABLED: {} {}; retrain the policy against "
                      "the current features".format(os.path.basename(path), _why),
                      flush=True)
                return None
            pg = {
                'features': pg_features,
                'feature_dim': feature_dim,
                'W1': np.asarray(payload['W1'], dtype=np.float64),
                'b1': np.asarray(payload['b1'], dtype=np.float64),
                'Wp_main': np.asarray(payload['Wp_main'], dtype=np.float64),
                'bp_main': np.asarray(payload['bp_main'], dtype=np.float64),
                'Wp_discard': np.asarray(payload['Wp_discard'], dtype=np.float64),
                'bp_discard': np.asarray(payload['bp_discard'], dtype=np.float64),
                'w_beta_a': np.asarray(payload['w_beta_a'], dtype=np.float64),
                'b_beta_a': float(payload.get('b_beta_a') or 0.0),
                'w_beta_b': np.asarray(payload['w_beta_b'], dtype=np.float64),
                'b_beta_b': float(payload.get('b_beta_b') or 0.0),
            }
            return pg
        except Exception:
            return None

    def _pg_select_action(self, round_state, active, legal_actions):
        pg = self._pg
        if pg is None:
            return None
        try:
            feats = pg['features'].extract_features(
                round_state, active, STARTING_STACK, BIG_BLIND, pg['feature_dim'])
            x = np.asarray(feats, dtype=np.float64)
            if x.shape[0] != pg['feature_dim']:
                return None
            h = np.maximum(0.0, pg['W1'] @ x + pg['b1'])
            if DiscardAction in legal_actions:
                logits = pg['Wp_discard'] @ h + pg['bp_discard']
                idx = int(np.argmax(logits))
                hand = round_state.hands[active]
                if idx < 0 or idx >= len(hand):
                    return None
                return DiscardAction(idx)
            cand = []
            if FoldAction in legal_actions:
                cand.append(0)
            if CallAction in legal_actions:
                cand.append(1)
            if CheckAction in legal_actions:
                cand.append(2)
            if RaiseAction in legal_actions:
                cand.append(3)
            if not cand:
                return None
            logits = pg['Wp_main'] @ h + pg['bp_main']
            best = cand[int(np.argmax(logits[cand]))]
            if best == 3:
                raw_a = float(pg['w_beta_a'] @ h + pg['b_beta_a'])
                raw_b = float(pg['w_beta_b'] @ h + pg['b_beta_b'])
                alpha = _pg_softplus(raw_a) + 1e-3
                beta = _pg_softplus(raw_b) + 1e-3
                frac = alpha / max(1e-9, alpha + beta)
                return _safe_raise(round_state, active, frac)
            if best == 0:
                return FoldAction()
            if best == 1:
                return CallAction()
            return CheckAction()
        except Exception:
            return None

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.board
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]
        continue_cost = opp_pip - my_pip
        my_contribution = STARTING_STACK - my_stack
        opp_contribution = STARTING_STACK - opp_stack
        pot = my_contribution + opp_contribution
        facing_raise = False
        raise_pressure = 0.0
        if continue_cost > 0 and my_pip > 0:
            facing_raise = True
            raise_pressure = float(continue_cost) / float(max(1.0, pot + continue_cost))
        try:
            last = getattr(self, '_last_seen_opp_pip', 0)
            if getattr(self, '_last_seen_street', None) != street:
                last = 0
                self._last_seen_street = street
            if opp_pip != last:
                self._this_round_opp_actions = getattr(self, '_this_round_opp_actions', 0) + 1
                if opp_pip > last:
                    self._this_round_opp_aggr = getattr(self, '_this_round_opp_aggr', 0) + 1
                    try:
                        if not hasattr(self, '_opp_aggr_streets'):
                            self._opp_aggr_streets = set()
                        self._opp_aggr_streets.add(int(street))
                        if continue_cost > 0:
                            p = float(continue_cost) / float(max(1.0, pot + continue_cost))
                            self._opp_max_pressure = max(
                                float(getattr(self, '_opp_max_pressure', 0.0) or 0.0), p)
                    except Exception:
                        pass
            if street == 6 and opp_pip > my_pip:
                self._opp_bet_river_this_round = True
            if street == 0:
                forced_blind = BIG_BLIND if active == 0 else SMALL_BLIND
                if opp_pip > forced_blind:
                    self._opp_vpip_this_round = True
                if opp_pip > BIG_BLIND:
                    self._opp_pfr_this_round = True
            self._last_seen_opp_pip = opp_pip
        except Exception:
            pass
        self._mccfr_sync_history(round_state, active)
        rounds_left = max(0, int(NUM_ROUNDS - game_state.round_num) + 1)
        if self._use_lock_in and float(game_state.bankroll) > 1.5 * float(rounds_left) + 1:
            if DiscardAction not in legal_actions:
                if continue_cost > 0:
                    return FoldAction()
                return CheckAction() if CheckAction in legal_actions else FoldAction()
        if self._use_pg and self._pg is not None:
            pg_action = self._pg_select_action(round_state, active, legal_actions)
            if pg_action is not None:
                return pg_action
        if self._mccfr_policy is not None and self._use_mccfr:
            mccfr_action = self._mccfr_select_action(round_state, active)
            try:
                stats = self._mccfr_stats
                stats['total'] = int(stats.get('total', 0) or 0) + 1
                if mccfr_action is None:
                    stats['misses'] = int(stats.get('misses', 0) or 0) + 1
                    mb = stats.get('miss_by_street', {})
                    mb[round_state.street] = int(mb.get(round_state.street, 0) or 0) + 1
                    stats['miss_by_street'] = mb
                    reason = self._mccfr_last_miss_reason or 'unknown'
                    mr = stats.get('miss_reason', {})
                    mr[reason] = int(mr.get(reason, 0) or 0) + 1
                    stats['miss_reason'] = mr
                else:
                    stats['hits'] = int(stats.get('hits', 0) or 0) + 1
                    if getattr(self, '_mccfr_last_fallback_used', False):
                        stats['fallback_hits'] = int(stats.get('fallback_hits', 0) or 0) + 1
                self._mccfr_stats = stats
            except Exception:
                pass
            if mccfr_action is not None:
                return mccfr_action
            if self._mccfr_strict:
                if CallAction in legal_actions and continue_cost > 0:
                    return CallAction()
                if CheckAction in legal_actions:
                    return CheckAction()
                if FoldAction in legal_actions:
                    return FoldAction()
        opp_bias = 0.0
        if continue_cost > 0:
            denom = float(max(1, pot + continue_cost))
            pressure = float(continue_cost) / denom
            opp_bias = max(0.0, min(1.0, 2.0 * pressure))
        fold_rate = 0.0
        try:
            fold_rate = float(self._opp_folds_to_our_raises) / float(max(1, self._our_raises))
        except Exception:
            fold_rate = 0.0
       
        try:
            avg_vpip = 0.0
            if self._opp_stats['vpip']:
                avg_vpip = sum(self._opp_stats['vpip']) / len(self._opp_stats['vpip'])
            opp_bias = max(0.0, min(1.0, opp_bias + 0.15 * (0.3 - avg_vpip)))
       
            avg_pfr = 0.0
            if self._opp_stats['pfr']:
                avg_pfr = sum(self._opp_stats['pfr']) / len(self._opp_stats['pfr'])
            opp_bias = max(0.0, min(1.0, opp_bias - 0.10 * max(0.0, avg_pfr - 0.25)))
       
            avg_fold_to_raise = 0.0
            if self._opp_stats['fold_to_raise']:
                avg_fold_to_raise = sum(self._opp_stats['fold_to_raise']) / len(self._opp_stats['fold_to_raise'])
            opp_bias = max(0.0, min(1.0, opp_bias + 0.20 * (avg_fold_to_raise - 0.5)))
        except Exception:
            pass
        in_position = False
        try:
            if street == 0:
                in_position = (active == 1)
            elif street >= 4:
                in_position = (active == 0)
            else:
                in_position = (active == 0)
        except Exception:
            in_position = False
        try:
            if street == 4 and in_position and continue_cost == 0 and opp_pip == 0:
                self._opp_flop_checked = True
        except Exception:
            pass
        try:
            opp_bias = max(0.0, min(1.0, float(opp_bias) + 0.25 * (fold_rate - 0.5)))
            if continue_cost > 0:
                opp_bias = max(0.0, min(1.0, opp_bias + (0.05 if not in_position else -0.02)))
        except Exception:
            opp_bias = max(0.0, min(1.0, float(opp_bias)))
        try:
            if self._use_range_track:
                streets_aggr = len(getattr(self, '_opp_aggr_streets', ()) or ())
                if streets_aggr > 1:
                    opp_bias = max(0.0, min(1.0, opp_bias + 0.08 * min(3, streets_aggr - 1)))
                mx = float(getattr(self, '_opp_max_pressure', 0.0) or 0.0)
                if mx >= 0.5 and streets_aggr >= 2:
                    opp_bias = max(0.0, min(1.0, opp_bias + 0.05))
        except Exception:
            pass
        try:
            if self._use_discard_read:
                opp_bias = max(0.0, min(1.0, opp_bias + self._opp_discard_bias(round_state, active)))
        except Exception:
            pass
        try:
            baseline = 0.25
            avg_aggr = float(getattr(self, '_avg_opp_aggr', 0.0) or 0.0)
            aggr_exploit = max(-1.0, min(1.0, avg_aggr - baseline))
            if aggr_exploit > 0.0:
                opp_bias = max(0.0, min(1.0, opp_bias * (1.0 - 0.35 * aggr_exploit)))
        except Exception:
            aggr_exploit = 0.0
        try:
            if facing_raise and street >= 4:
                opp_bias = max(0.0, min(1.0, opp_bias + 0.12 + 0.25 * raise_pressure))
        except Exception:
            pass
        try:
            if self._self_strategy_history:
                avg_self = float(sum(x.get('delta', 0.0) for x in self._self_strategy_history)) / float(len(self._self_strategy_history))
            else:
                avg_self = 0.0
            self._avg_self_perf = avg_self
        except Exception:
            self._avg_self_perf = 0.0
        tex = _board_texture_key(board_cards)
        paired_board = (tex[1] >= 2 or tex[2] >= 2)
        draw_heavy = (tex[6] >= 3 or (len(tex) > 9 and tex[9] >= 4)) and len(board_cards) < 5
        board_completes_draw = self._board_completes_draw(board_cards)
        dry_board = (tex[6] <= 1 and (len(tex) <= 9 or tex[9] <= 3) and tex[1] == 1)
        print(f"Hand: {my_cards}")
        print(f"Board: {board_cards}")
        if DiscardAction in legal_actions:
            iters = 120
            do_second_pass = True
            if self._mccfr_policy is not None and self._use_mccfr:
                iters = min(iters, self._mccfr_discard_iters)
                do_second_pass = False
            iters_per = max(25, iters // 2)
            tightness = self._estimate_opp_tightness()
            base_bias = 0.08 if active == 1 else 0.03
            base_bias = min(1.0, base_bias + 0.30 * fold_rate)
            discard_bias = max(0.0, min(1.0, opp_bias + base_bias + 0.12 * max(0.0, tightness)))
            equity_by_discard = []
            for discard_idx in (0, 1, 2):
                kept = [my_cards[i] for i in (0, 1, 2) if i != discard_idx]
                board_after = list(board_cards) + [my_cards[discard_idx]]
                equity = self.calculate_equity(kept, board_after, iters=iters_per, opp_bias=discard_bias)
                equity_by_discard.append((equity, discard_idx))
            equity_by_discard.sort(reverse=True)
            if len(equity_by_discard) >= 2:
                gap = equity_by_discard[0][0] - equity_by_discard[1][0]
                if gap < 0.03 and do_second_pass:
                    top = equity_by_discard[:2]
                    equity_by_discard = []
                    for _, discard_idx in top:
                        kept = [my_cards[i] for i in (0, 1, 2) if i != discard_idx]
                        board_after = list(board_cards) + [my_cards[discard_idx]]
                        equity = self.calculate_equity(kept, board_after, iters=iters, opp_bias=discard_bias)
                        equity_by_discard.append((equity, discard_idx))
                    equity_by_discard.sort(reverse=True)
            best_discard = equity_by_discard[0][1] if equity_by_discard else 0
            try:
                self._this_round_our_discards = getattr(self, '_this_round_our_discards', 0) + 1
            except Exception:
                pass
            return DiscardAction(best_discard)
          
        if self.kept_cards is None and len(my_cards) == 2:
            self.kept_cards = my_cards
          
        clock = 0.0
        try:
            clock = float(getattr(game_state, 'game_clock', 0.0) or 0.0)
        except Exception:
            clock = 0.0
        if clock <= 2.5:
            iters = 25
        elif clock <= 6.0:
            iters = 50
        else:
            iters = 120 if street == 0 else 100 if street == 3 else 80
        if self._iter_scale != 1.0 and clock > 6.0:
            iters = max(10, int(round(iters * self._iter_scale)))
        if self._mccfr_policy is not None and self._use_mccfr:
            iters = min(int(iters), int(self._mccfr_fallback_iters))
          
        is_river = (self._use_river_logic and street == 6
                    and self.kept_cards is not None and len(board_cards) == 6)
        river_bet_mode = is_river and self._use_river_betting
        equity = 0.5
        if not river_bet_mode:
            if self.kept_cards is not None and len(board_cards) >= 4:
                equity = self.calculate_equity(self.kept_cards, board_cards, iters=iters, opp_bias=opp_bias)
            elif street == 0:
                if self._mccfr_policy is not None and self._use_mccfr:
                    preflop_iters = min(int(iters), int(self._mccfr_preflop_iters))
                else:
                    preflop_iters = min(120, int(iters))
                equity = self.calculate_hand_potential(my_cards, [], iters=preflop_iters, opp_bias=opp_bias)
        print(f"Street: {street}, Equity: {equity:.2f}, Fold Rate: {fold_rate:.2f}, In Position: {in_position}, Pot: {pot}, Continue Cost: {continue_cost}")
        draw_heavy = False
        board_completes_draw = False
        try:
            tex = _board_texture_key(board_cards)
            max_suit, best_run = tex[6], tex[9]
            draw_heavy = (max_suit >= 3 or best_run >= 4) and len(board_cards) < 5
            if len(board_cards) == 6:
                prev_tex = _board_texture_key(board_cards[:-1])
                board_completes_draw = (max_suit >= 4 and prev_tex[6] < 4) or (best_run >= 5 and prev_tex[9] < 5)
        except Exception:
            pass
       
        if continue_cost > 0:
            if continue_cost > my_stack:
                continue_cost = my_stack
              
            pot_after_call = my_contribution + opp_contribution + continue_cost
            required_equity = continue_cost / max(1.0, (pot + continue_cost))
            try:
                required_equity = max(0.0, min(1.0, required_equity - 0.12 * max(0.0, float(getattr(self, '_avg_opp_aggr', 0.0) or 0.0) - 0.25)))
            except Exception:
                pass
            spr = my_stack / max(1.0, pot_after_call) if pot_after_call > 0 else 10.0
            implied_mult = 1.0
            if draw_heavy and spr > 6.0 and fold_rate < 0.4:
                implied_mult = 1.0 + 0.025 * min(4.0, spr - 6.0)
                cap = {4: 1.05, 5: 1.03}.get(street, 1.0 if street >= 6 else 1.10)
                implied_mult = min(cap, implied_mult)
            equity *= implied_mult
            alpha = continue_cost / max(1.0, pot + continue_cost)
            mdf = alpha / (alpha + 1.0) if alpha > 0 else 0.5
            required_equity = max(required_equity, mdf * 0.4) + 0.03
            required_equity = min(1.0, required_equity)
            if street == 0 and not in_position:
                required_equity += 0.10
                if fold_rate < 0.4:
                    required_equity += 0.05
                if equity < 0.45 and fold_rate < 0.5:
                    print("Action: Fold (preflop OOP tighten)")
                    return FoldAction()
            if facing_raise and street >= 4:
                required_equity += 0.06 + 0.10 * raise_pressure
                if paired_board or board_completes_draw:
                    required_equity += 0.06
                if raise_pressure >= 0.45:
                    required_equity = max(required_equity, 0.62)
                required_equity = min(1.0, required_equity)
            print(f"Required Equity: {required_equity:.2f}, MDF: {mdf:.2f}, Implied Mult: {implied_mult:.2f}, SPR: {spr:.2f}")
            if street == 4:
                if opp_pip > my_pip and self._opp_flop_checked:
                    required_equity += 0.15
                    opp_bias += 0.2
                if facing_raise and raise_pressure >= 0.35 and (paired_board or board_completes_draw):
                    required_equity = max(required_equity, 0.58)
            if is_river:
                pressure = float(continue_cost) / float(max(1.0, pot + continue_cost))
                q = self._river_quantile_from_pressure(pressure, int(my_pip))
                try:
                    avg_aggr = float(getattr(self, '_avg_opp_aggr', 0.0) or 0.0)
                    if avg_aggr > 0.35:
                        q = max(0.80, q - 0.03)
                    elif avg_aggr < 0.18:
                        q = min(0.99, q + 0.03)
                    if board_completes_draw:
                        q = min(0.99, q + 0.03)
                    if facing_raise:
                        q = min(0.99, q + 0.05 + (0.04 if pressure >= 0.40 else 0.0))
                except Exception:
                    pass
                equity_vs_value = self._river_equity_vs_top_quantile(self.kept_cards, board_cards, q)
                river_required = float(continue_cost) / float(max(1.0, pot + 2.0 * continue_cost))
                min_bluff = self._min_bluff_freq_for_call(river_required, equity_vs_value)
                max_bluff = self._river_max_bluff(
                    int(my_pip), pressure, facing_raise, board_completes_draw, paired_board)
                if self._use_blockers:
                    blk = self._blocker_strength(self.kept_cards, board_cards, q)
                    max_bluff = max(0.02, min(0.85, max_bluff * (0.75 + 0.25 * blk)))
                generic_margin = float(equity) - float(river_required)
                gate_folds = min_bluff > max_bluff
                if gate_folds and generic_margin <= self.RIVER_GATE_MARGIN:
                    try:
                        self._this_round_our_folds = getattr(self, '_this_round_our_folds', 0) + 1
                    except Exception:
                        pass
                    print(f"Action: Fold (river bluff gate, margin={generic_margin:+.3f})")
                    return FoldAction()
                if RaiseAction in legal_actions and my_pip == 0:
                    eq_vs_top10 = self._river_equity_vs_top_quantile(self.kept_cards, board_cards, 0.90)
                    if eq_vs_top10 >= 0.75 and pressure <= 0.30:
                        min_raise, max_raise = round_state.raise_bounds()
                        size = 0.7 if eq_vs_top10 >= 0.85 else 0.6
                        raise_amount = int(my_pip + continue_cost + pot_after_call * size)
                        raise_amount = min(max_raise, max(min_raise, raise_amount))
                        try:
                            self._this_round_our_raises = getattr(self, '_this_round_our_raises', 0) + 1
                            if equity >= 0.65:
                                self._this_round_our_valuebets = getattr(self, '_this_round_our_valuebets', 0) + 1
                            else:
                                self._this_round_our_bluffs = getattr(self, '_this_round_our_bluffs', 0) + 1
                        except Exception:
                            pass
                        self._our_raises += 1
                        self._last_action_was_raise = True
                        print(f"Action: Raise (river value, amount={raise_amount})")
                        return RaiseAction(raise_amount)
                print("Action: Call (river)")
                self._this_round_our_calls = getattr(self, '_this_round_our_calls', 0) + 1
                return CallAction()
           
            if equity > required_equity:
                opp_equity_approx = 1 - equity
                if RaiseAction in legal_actions:
                    min_raise, max_raise = round_state.raise_bounds()
                    if equity >= 0.65:
                        size = 0.8
                    elif equity >= 0.55:
                        size = 0.6
                    else:
                        size = 0.3
                    raise_amount = int(my_pip + continue_cost + pot_after_call * size)
                    raise_amount = min(max_raise, max(min_raise, raise_amount))
                    raise_cost = raise_amount - my_pip - continue_cost
                    p_fold = 0.4 * (1 - opp_equity_approx)
                    ev_if_fold = opp_contribution
                    ev_if_call = equity * (opp_contribution + (raise_amount - opp_pip)) - (1-equity) * (my_contribution + continue_cost + raise_cost)
                    ev_raise = p_fold * ev_if_fold + (1 - p_fold) * ev_if_call
                    ev_call = equity * pot_after_call - (1-equity) * (my_contribution + continue_cost)
                    print(f"EV Raise: {ev_raise:.2f}, EV Call: {ev_call:.2f}")
                  
                    if ev_raise > ev_call * 1.1 and not river_bet_mode and fold_rate > 0.35:
                        try:
                            self._this_round_our_raises = getattr(self, '_this_round_our_raises', 0) + 1
                            if equity >= 0.60:
                                self._this_round_our_valuebets = getattr(self, '_this_round_our_valuebets', 0) + 1
                            else:
                                self._this_round_our_bluffs = getattr(self, '_this_round_our_bluffs', 0) + 1
                        except Exception:
                            pass
                        self._our_raises += 1
                        self._last_action_was_raise = True
                        print(f"Action: Raise (amount={raise_amount})")
                        return RaiseAction(raise_amount)
                print("Action: Call")
                self._this_round_our_calls = getattr(self, '_this_round_our_calls', 0) + 1
                return CallAction()
            else:
                try:
                    self._this_round_our_folds = getattr(self, '_this_round_our_folds', 0) + 1
                except Exception:
                    pass
                print("Action: Fold")
                return FoldAction()
        else:
            if (self._use_check_raise and RaiseAction in legal_actions
                    and not in_position and street >= 4 and continue_cost == 0):
                try:
                    aggr = float(getattr(self, '_avg_opp_aggr', 0.0) or 0.0)
                    have_reads = len(getattr(self, '_opp_aggr_history', ()) or ()) >= 5
                    if (have_reads and aggr >= self.CHECK_RAISE_MIN_AGGR
                            and equity >= self.CHECK_RAISE_MIN_EQUITY
                            and random.random() < self.CHECK_RAISE_FREQ):
                        self._this_round_our_checks = getattr(self, '_this_round_our_checks', 0) + 1
                        print(f"Action: Check (check-raise trap, eq={equity:.2f}, opp_aggr={aggr:.2f})")
                        return CheckAction()
                except Exception:
                    pass
            if RaiseAction in legal_actions:
                if river_bet_mode:
                    q = 0.80
                    try:
                        avg_aggr = float(getattr(self, '_avg_opp_aggr', 0.0) or 0.0)
                        if avg_aggr < 0.18:
                            q += 0.05
                        elif avg_aggr > 0.35:
                            q -= 0.03
                    except Exception:
                        pass
                    if board_completes_draw:
                        q += 0.05
                    q = max(0.70, min(0.97, q))
                    eq_uniform = self._river_equity_vs_top_quantile(self.kept_cards, board_cards, 0.0)
                    eq_vs_range = self._river_equity_vs_top_quantile(self.kept_cards, board_cards, q)
                    eq_vs_top10 = self._river_equity_vs_top_quantile(self.kept_cards, board_cards, min(0.98, q + 0.08))
                    print(f"River Equity Uniform: {eq_uniform:.2f}, vs RangeQ{q:.2f}: {eq_vs_range:.2f}, vs Top10: {eq_vs_top10:.2f}")
                    min_eq_uniform = 0.58
                    min_eq_range = 0.50
                    min_eq_top = 0.40
                    if paired_board or board_completes_draw:
                        min_eq_uniform = 0.62
                        min_eq_range = 0.55
                        min_eq_top = 0.48
                    if eq_uniform >= min_eq_uniform and eq_vs_range >= min_eq_range and eq_vs_top10 >= min_eq_top:
                        min_raise, max_raise = round_state.raise_bounds()
                        size = 0.65 if board_completes_draw else (0.75 if eq_uniform < 0.75 else 1.0)
                        raise_amount = int(my_pip + pot * size)
                        raise_amount = min(max_raise, max(min_raise, raise_amount))
                        try:
                            self._this_round_our_raises = getattr(self, '_this_round_our_raises', 0) + 1
                            if eq_uniform >= 0.60:
                                self._this_round_our_valuebets = getattr(self, '_this_round_our_valuebets', 0) + 1
                            else:
                                self._this_round_our_bluffs = getattr(self, '_this_round_our_bluffs', 0) + 1
                        except Exception:
                            pass
                        self._our_raises += 1
                        self._last_action_was_raise = True
                        print(f"Action: Raise (river bet, amount={raise_amount})")
                        return RaiseAction(raise_amount)
                    try:
                        self._this_round_our_checks = getattr(self, '_this_round_our_checks', 0) + 1
                    except Exception:
                        pass
                    print("Action: Check (river)")
                    return CheckAction()
                try:
                    bluff_mult = 1.0 - 0.8 * max(0.0, float(getattr(self, '_avg_opp_aggr', 0.0) or 0.0) - 0.25)
                    bluff_mult = max(0.05, min(1.0, bluff_mult))
                    avg_self = float(getattr(self, '_avg_self_perf', 0.0) or 0.0)
                    if avg_self < 0.0:
                        perf_mult = 1.0 - min(0.6, (-avg_self) / 10.0)
                        bluff_mult = max(0.05, bluff_mult * perf_mult)
                except Exception:
                    bluff_mult = 1.0
                bluff_prob = 0.10 * bluff_mult * max(0.3, fold_rate)
                if street == 0 and not in_position:
                    bluff_prob *= 0.5
                is_bluff = False
                if equity < 0.35 and random.random() < bluff_prob:
                    is_bluff = True
                elif equity > 0.62:
                    is_bluff = False
                else:
                    is_bluff = None
                if is_bluff is not None:
                    min_raise, max_raise = round_state.raise_bounds()
                    size = 0.33 if is_bluff else 0.65 if equity > 0.70 else 0.5
                    raise_amount = int(my_pip + pot * size)
                    raise_amount = min(max_raise, max(min_raise, raise_amount))
                    try:
                        self._this_round_our_raises = getattr(self, '_this_round_our_raises', 0) + 1
                        if not is_bluff:
                            self._this_round_our_valuebets = getattr(self, '_this_round_our_valuebets', 0) + 1
                        else:
                            self._this_round_our_bluffs = getattr(self, '_this_round_our_bluffs', 0) + 1
                    except Exception:
                        pass
                    self._our_raises += 1
                    self._last_action_was_raise = True
                    print(f"Action: Raise (bet, amount={raise_amount}, bluff={is_bluff})")
                    return RaiseAction(raise_amount)
            try:
                self._this_round_our_checks = getattr(self, '_this_round_our_checks', 0) + 1
            except Exception:
                pass
            print("Action: Check")
            return CheckAction()
class PoolOpponent(Bot):

    def __init__(self):
        seed = os.environ.get('POOL_SEED', '').strip()
        if seed:
            try:
                random.seed(int(seed))
            except Exception:
                random.seed(seed)
            print(f"[pool] POOL_SEED={seed}")

        force = os.environ.get('POOL_OPPONENT', '').strip()
        chosen_cls = None
        if force:
            if force.lower() not in {'selfplay', 'selfplaysnapshot', 'snapshot', 'self'}:
                for cls in OPPONENT_POOL:
                    if cls.__name__.lower() == force.lower():
                        chosen_cls = cls
                        break
                if chosen_cls is not None:
                    print(f"[pool] POOL_OPPONENT={force} -> using {chosen_cls.__name__}")
                else:
                    print(f"[pool] POOL_OPPONENT={force} did not match pool; selecting randomly")
            else:
                print(f"[pool] POOL_OPPONENT={force} requested, but self-play is disabled here; selecting randomly")

        if chosen_cls is None:
            chosen_cls = random.choice(OPPONENT_POOL)
            print(f"[pool] Selected opponent: {chosen_cls.__name__}")

        self.impl = chosen_cls()

    def handle_new_round(self, game_state, round_state, active):
        fn = getattr(self.impl, 'handle_new_round', None)
        if callable(fn):
            try:
                fn(game_state, round_state, active)
            except Exception as e:
                try:
                    import traceback
                    print(f"[pool] handle_new_round error in {type(self.impl).__name__}: {e}")
                    traceback.print_exc()
                except Exception:
                    pass

    def handle_round_over(self, game_state, terminal_state, active):
        fn = getattr(self.impl, 'handle_round_over', None)
        if callable(fn):
            try:
                fn(game_state, terminal_state, active)
            except Exception as e:
                try:
                    import traceback
                    print(f"[pool] handle_round_over error in {type(self.impl).__name__}: {e}")
                    traceback.print_exc()
                except Exception:
                    pass

    def get_action(self, game_state, round_state, active):
        try:
            return self.impl.get_action(game_state, round_state, active)
        except Exception as e:
            try:
                import traceback
                print(f"[pool] get_action error in {type(self.impl).__name__}: {e}")
                traceback.print_exc()
            except Exception:
                pass

            legal = round_state.legal_actions()
            if CheckAction in legal:
                return CheckAction()
            if CallAction in legal:
                return CallAction()
            if DiscardAction in legal:
                try:
                    hand = round_state.hands[active]
                    return DiscardAction(0 if hand else 0)
                except Exception:
                    return DiscardAction(0)
            if FoldAction in legal:
                return FoldAction()
            return CheckAction()
OPPONENT_POOL = [
    GoodBot,
]

def _parse_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, '').strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)
if __name__ == '__main__':
    run_bot(PoolOpponent(), parse_args())
