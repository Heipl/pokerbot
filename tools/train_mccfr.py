import argparse
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import pkrbot

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "mccfr_bot"))

from skeleton.states import STARTING_STACK, BIG_BLIND, SMALL_BLIND
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
    info_key,
    three_card_shape,
    get_raise_fraction,
    select_bucket_count,
)


FOLD = 0
CALL_CHECK = 1
RAISE_XSMALL = 2
RAISE_SMALL = 3
RAISE_MED = 4
RAISE_LARGE = 5
RAISE_OVER = 6
# Floor for the opponent-reach multiplier. Regret matching gives exactly 0.0 to
# any action with non-positive regret, and reach_opp is multiplied by strat[a]
# at every node. A pool bot routinely plays actions our model calls impossible,
# so without a floor reach_opp hits 0 and the rest of the traversal teaches
# nothing.
MIN_OPP_REACH = 1e-3

RAISE_ALLIN = 7
DISCARD_0 = 8
DISCARD_1 = 9
DISCARD_2 = 10
NUM_ACTIONS = len(MCCFR_ACTIONS)


@dataclass
class TerminalState:
    deltas: list
    previous_state: "TossState"


@dataclass
class TossState:
    button: int
    street: int
    pips: list
    stacks: list
    hands: list
    board: list
    deck: pkrbot.Deck
    raise_count: int
    last_aggressor: int
    last_bet_bucket: int

    def clone(self):
        return TossState(
            self.button,
            self.street,
            list(self.pips),
            list(self.stacks),
            [list(h) for h in self.hands],
            list(self.board),
            self.deck,
            self.raise_count,
            self.last_aggressor,
            self.last_bet_bucket,
        )

    def active(self):
        return self.button % 2

    def pot(self):
        return 2 * STARTING_STACK - self.stacks[0] - self.stacks[1]

    def continue_cost(self):
        a = self.active()
        return self.pips[1 - a] - self.pips[a]

    def raise_bounds(self):
        active = self.active()
        continue_cost = self.continue_cost()
        max_contribution = min(self.stacks[active], self.stacks[1 - active] + continue_cost)
        min_contribution = min(max_contribution, continue_cost + max(continue_cost, BIG_BLIND))
        return (self.pips[active] + min_contribution, self.pips[active] + max_contribution)

    def legal_action_mask(self):
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        active = self.active()
        continue_cost = self.continue_cost()
        if self.street in (2, 3):
            if active != (self.street % 2):
                if len(self.hands[active]) >= 1:
                    mask[DISCARD_0] = 1.0
                if len(self.hands[active]) >= 2:
                    mask[DISCARD_1] = 1.0
                if len(self.hands[active]) >= 3:
                    mask[DISCARD_2] = 1.0
                return mask
            mask[CALL_CHECK] = 1.0
            return mask
        if continue_cost == 0:
            mask[CALL_CHECK] = 1.0
            bets_forbidden = (self.stacks[0] == 0 or self.stacks[1] == 0)
            if not bets_forbidden:
                for idx in MCCFR_ALLOWED_RAISES:
                    mask[idx] = 1.0
            return mask
        mask[CALL_CHECK] = 1.0
        mask[FOLD] = 1.0
        raises_forbidden = (continue_cost == self.stacks[active] or self.stacks[1 - active] == 0)
        if not raises_forbidden:
            for idx in MCCFR_ALLOWED_RAISES:
                mask[idx] = 1.0
        return mask

    def proceed_street(self):
        if self.street == 6:
            return self.showdown()
        if self.street == 0:
            new_street = 2
            button = 1
            board = list(self.board)
            board.extend(self.deck.peek(new_street))
            return TossState(
                button,
                new_street,
                [0, 0],
                self.stacks,
                self.hands,
                board,
                self.deck,
                0,
                -1,
                0,
            )
        if self.street == 2:
            return TossState(0, 3, [0, 0], self.stacks, self.hands, self.board, self.deck, 0, -1, 0)
        if self.street == 3:
            return TossState(1, 4, [0, 0], self.stacks, self.hands, self.board, self.deck, 0, -1, 0)
        new_street = self.street + 1
        board = list(self.board)
        try:
            board.append(self.deck.peek(new_street - 1)[new_street - 2])
        except Exception:
            pass
        return TossState(1, new_street, [0, 0], self.stacks, self.hands, board, self.deck, 0, -1, 0)

    def proceed(self, action_idx):
        active = self.active()
        if action_idx in (DISCARD_0, DISCARD_1, DISCARD_2):
            idx = action_idx - DISCARD_0
            state = self.clone()
            if idx < len(state.hands[active]):
                state.board.append(state.hands[active].pop(idx))
            state.button = (1 - active) % 2
            return state
        if action_idx == FOLD:
            delta = self.stacks[0] - STARTING_STACK if active == 0 else STARTING_STACK - self.stacks[1]
            return TerminalState([delta, -delta], self)
        if action_idx == CALL_CHECK:
            continue_cost = self.continue_cost()
            if continue_cost == 0:
                if (self.street == 0 and self.button > 0) or self.button > 1 or self.street in (2, 3):
                    return self.proceed_street()
                state = self.clone()
                state.button += 1
                return state
            if self.button == 0:
                return TossState(
                    1,
                    0,
                    [BIG_BLIND] * 2,
                    [STARTING_STACK - BIG_BLIND] * 2,
                    self.hands,
                    self.board,
                    self.deck,
                    self.raise_count,
                    self.last_aggressor,
                    self.last_bet_bucket,
                )
            state = self.clone()
            contribution = state.pips[1 - active] - state.pips[active]
            state.stacks[active] -= contribution
            state.pips[active] += contribution
            state.button += 1
            return state.proceed_street()
        if action_idx in (RAISE_XSMALL, RAISE_SMALL, RAISE_MED, RAISE_LARGE, RAISE_OVER, RAISE_ALLIN):
            min_raise, max_raise = self.raise_bounds()
            if action_idx == RAISE_ALLIN:
                amount = max_raise
            else:
                frac = get_raise_fraction(self.street, action_idx)
                pot_after_call = self.pot() + max(0, self.continue_cost())
                base = max(1.0, pot_after_call)
                amount = int(self.pips[active] + max(0, self.continue_cost()) + base * frac)
            amount = max(min_raise, min(max_raise, amount))
            state = self.clone()
            contribution = amount - state.pips[active]
            state.stacks[active] -= contribution
            state.pips[active] += contribution
            state.button += 1
            state.raise_count = min(3, state.raise_count + 1)
            state.last_aggressor = active
            if action_idx == RAISE_ALLIN:
                state.last_bet_bucket = 6
            else:
                state.last_bet_bucket = int(action_idx - RAISE_XSMALL + 1)
            return state
        return self

    def showdown(self):
        board = list(self.board)
        try:
            s0 = int(pkrbot.evaluate(board + self.hands[0]))
            s1 = int(pkrbot.evaluate(board + self.hands[1]))
        except Exception:
            s0 = 0
            s1 = 0
        if s0 > s1:
            delta = STARTING_STACK - self.stacks[1]
        elif s1 > s0:
            delta = self.stacks[0] - STARTING_STACK
        else:
            delta = 0
        return TerminalState([int(delta), -int(delta)], self)


class PoolOpponentAdapter:
    '''Let a bot from opponent_pool act at the opponent nodes of a traversal.

    WHY. This trainer was pure self-play: at opponent nodes it sampled from its
    OWN regret-matched strategy, so the table only ever converged to an
    equilibrium of the ABSTRACTED game against itself. It never observed how any
    real opponent bets. The measured consequence, against BluffHeavyBot:

        traversals   turn call%   street5 call%   chips
        4.45M          18.3%          36.3%      -14,469
        5.30M          23.4%          45.8%      -20,431
        heuristic       5.0%           9.0%       +2,496

    More traversals sharpened the policy TOWARD calling and lost more, because
    self-play gave it no reason to learn that this opponent's bets are worth
    folding to. That is not a coverage problem and no traversal budget fixes it.

    Training some fraction of traversals against real pool bots computes a
    (partial) best response to them instead. Mixing with self-play keeps the
    equilibrium pressure that stops it overfitting to one opponent -- the same
    trade-off tools/train_goodbot_pg.py makes with --train-vs-pool-prob.
    '''

    def __init__(self, name: str, seed: int = 0):
        import importlib.util
        self.name = name
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # engine.py is at the project root, not necessarily on the path here.
        if root not in sys.path:
            sys.path.insert(0, root)
        self._engine = importlib.import_module("engine")
        path = os.path.join(root, "opponent_pool", "player.py")
        spec = importlib.util.spec_from_file_location("mccfr_pool_player", path)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, os.path.join(root, "opponent_pool"))
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.path.pop(0)
        for nm in ("FoldAction", "CallAction", "CheckAction", "RaiseAction", "DiscardAction"):
            setattr(mod, nm, getattr(self._engine, nm))
        for nm in ("STARTING_STACK", "BIG_BLIND", "SMALL_BLIND"):
            setattr(mod, nm, int(getattr(self._engine, nm)))
        setattr(mod, "NUM_ROUNDS", 1000)
        cls = getattr(mod, name, None)
        if cls is None:
            raise SystemExit(f"ERROR: no bot named {name!r} in opponent_pool/player.py")
        self._mod = mod
        self.bot = cls()

        class _GS:
            bankroll = 0
            game_clock = 1e9
            round_num = 1
        self._gs = _GS()

    def new_hand(self, hand_no: int) -> None:
        self._gs.round_num = int(hand_no)
        try:
            self.bot.handle_new_round(self._gs, None, 1)
        except Exception:
            pass

    def action_index(self, state, legal_mask):
        '''Ask the pool bot to act; return an index into the trainer's actions.

        Returns None if anything goes wrong, so the caller falls back to
        self-play sampling rather than silently biasing the traversal.
        '''
        eng = self._engine
        active = state.active()
        try:
            rs = eng.RoundState(state.button, state.street, list(state.pips),
                                list(state.stacks), [list(h) for h in state.hands],
                                state.deck, list(state.board), None)
            act = self.bot.get_action(self._gs, rs, active)
        except Exception:
            return None

        name = type(act).__name__
        if name == "FoldAction":
            idx = FOLD
        elif name in ("CallAction", "CheckAction"):
            idx = CALL_CHECK
        elif name == "DiscardAction":
            idx = DISCARD_0 + int(getattr(act, "card", 0))
        elif name == "RaiseAction":
            # Invert TossState.proceed's bucket -> amount formula exactly:
            #   amount = pips[active] + cc + max(1, pot() + cc) * frac
            # Dropping cc turns every re-raise into a larger bucket than the bot
            # actually played.
            cc = max(0, int(state.continue_cost()))
            denom = max(1.0, float(state.pot()) + float(cc))
            extra = (float(getattr(act, "amount", 0))
                     - float(state.pips[active]) - float(cc))
            frac = max(0.0, extra) / denom
            frac_map = MCCFR_RAISE_FRACTIONS.get(state.street)
            if not isinstance(frac_map, dict):
                frac_map = MCCFR_RAISE_FRACTIONS.get("default", {})
            # Snap only to legal buckets: the fraction table is keyed
            # {2..6} but MCCFR_ALLOWED_RAISES is (3, 4, 6, 7), so 2 and 5 are
            # snap targets that can never be played and an ordinary re-raise
            # would fall back to self-play. Bucket 7 (all-in) has no fraction,
            # so a raise at the table maximum maps onto it directly.
            try:
                _, max_raise = state.raise_bounds()
            except Exception:
                max_raise = None
            if (max_raise is not None
                    and int(getattr(act, "amount", 0)) >= int(max_raise)
                    and RAISE_ALLIN < len(legal_mask)
                    and legal_mask[RAISE_ALLIN] > 0):
                idx = RAISE_ALLIN
            else:
                best, best_d = None, None
                for a_idx, f in frac_map.items():
                    a_idx = int(a_idx)
                    if a_idx >= len(legal_mask) or legal_mask[a_idx] <= 0:
                        continue
                    d = abs(float(f) - frac)
                    if best_d is None or d < best_d:
                        best, best_d = a_idx, d
                idx = best if best is not None else CALL_CHECK
        else:
            return None

        if idx is None or idx >= len(legal_mask) or legal_mask[idx] <= 0:
            # Not representable in the abstraction; fall back, don't force it.
            return None
        return int(idx)


def regret_matching(regrets, legal_mask):
    positive = np.maximum(regrets, 0.0) * legal_mask
    denom = positive.sum()
    if denom > 1e-12:
        return positive / denom
    legal = legal_mask.sum()
    if legal <= 0:
        return np.zeros_like(regrets)
    return legal_mask / legal


class MCCFRTrainer:
    def __init__(
        self,
        score_ranges,
        hand_buckets=HAND_BUCKETS,
        board_buckets=BOARD_BUCKETS,
        pot_buckets=POT_BB_BUCKETS,
        spr_buckets=SPR_BUCKETS,
        pressure_buckets=PRESSURE_BUCKETS,
        epsilon_start=0.12,
        epsilon_end=0.02,
        epsilon_decay=300000,
        epsilon_postflop_mult=1.25,
        cache_max=250000,
    ):
        self.regrets = {}
        self.strategy_sum = {}
        self.score_ranges = score_ranges
        self.hand_buckets = hand_buckets
        self.board_buckets = board_buckets
        self.traversals = 0
        self.pot_buckets = pot_buckets
        self.spr_buckets = spr_buckets
        self.pressure_buckets = pressure_buckets
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_postflop_mult = float(epsilon_postflop_mult)
        self._hand_bucket_cache = {}
        self._board_bucket_cache = {}
        self._shape_cache = {}
        self._cache_max = max(0, int(cache_max))

    def _epsilon(self, street):
        if self.epsilon_decay <= 0:
            eps = self.epsilon_end
        else:
            progress = min(1.0, float(self.traversals) / float(self.epsilon_decay))
            eps = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress
        if street >= 4:
            eps = min(0.5, eps * self.epsilon_postflop_mult)
        return max(0.0, min(0.9, eps))

    def _info_key(self, state):
        active = state.active()
        pot = state.pot()
        pot_bb = pot / float(BIG_BLIND)
        spr = state.stacks[active] / float(max(1.0, pot))
        continue_cost = state.continue_cost()
        pressure = float(continue_cost) / float(max(1.0, pot + continue_cost))
        facing_raise = bool(continue_cost > 0 and state.pips[active] > 0)
        raise_count_bucket = min(2, int(state.raise_count))
        if state.last_aggressor < 0:
            last_aggressor = 0
        else:
            last_aggressor = 1 if state.last_aggressor == active else 2
        last_bet_bucket = int(state.last_bet_bucket)
        default_hand_buckets = select_bucket_count(HAND_BUCKETS, state.street, 8)
        default_board_buckets = select_bucket_count(BOARD_BUCKETS, state.street, 32)
        hand_bucket_count = select_bucket_count(self.hand_buckets, state.street, default_hand_buckets)
        board_bucket_count = select_bucket_count(self.board_buckets, state.street, default_board_buckets)
        cards = list(state.hands[active])
        board = list(state.board)
        try:
            cards_key = tuple(sorted(str(c) for c in cards))
        except Exception:
            cards_key = tuple(str(c) for c in cards)
        try:
            board_key = tuple(sorted(str(c) for c in board))
        except Exception:
            board_key = tuple(str(c) for c in board)

        if len(cards) == 3:
            hb_key = ("3", cards_key, hand_bucket_count)
        elif len(cards) == 2:
            hb_key = ("2", cards_key, board_key, hand_bucket_count)
        else:
            hb_key = ("x", cards_key, board_key, hand_bucket_count)
        hand_bucket = self._hand_bucket_cache.get(hb_key)
        if hand_bucket is None:
            hand_bucket = hand_strength_bucket(
                cards,
                board,
                self.score_ranges,
                bucket_count=hand_bucket_count,
            )
            self._hand_bucket_cache[hb_key] = hand_bucket
            if self._cache_max and len(self._hand_bucket_cache) > self._cache_max:
                self._hand_bucket_cache.clear()

        bb_key = (board_key, board_bucket_count)
        board_bucket = self._board_bucket_cache.get(bb_key)
        if board_bucket is None:
            board_bucket = board_texture_bucket(board, bucket_count=board_bucket_count)
            self._board_bucket_cache[bb_key] = board_bucket
            if self._cache_max and len(self._board_bucket_cache) > self._cache_max:
                self._board_bucket_cache.clear()

        shape_key = (cards_key, len(cards))
        shape = self._shape_cache.get(shape_key)
        if shape is None:
            shape = three_card_shape(cards)
            self._shape_cache[shape_key] = shape
            if self._cache_max and len(self._shape_cache) > self._cache_max:
                self._shape_cache.clear()
        return info_key(
            state.street,
            active,
            pot_bb,
            spr,
            pressure,
            facing_raise,
            raise_count_bucket,
            last_aggressor,
            last_bet_bucket,
            hand_bucket,
            board_bucket,
            shape,
            pot_buckets=self.pot_buckets,
            spr_buckets=self.spr_buckets,
            pressure_buckets=self.pressure_buckets,
        )

    def traverse(self, state, traverser, reach_trav, reach_opp):
        if isinstance(state, TerminalState):
            return float(state.deltas[traverser])
        if state.street > 6:
            terminal = state.showdown()
            return float(terminal.deltas[traverser])
        active = state.active()
        info = self._info_key(state)
        legal_mask = state.legal_action_mask()
        regrets = self.regrets.get(info)
        if regrets is None:
            regrets = np.zeros(NUM_ACTIONS, dtype=np.float32)
            self.regrets[info] = regrets
        strat = regret_matching(regrets, legal_mask)
        strat_sum = self.strategy_sum.get(info)
        if strat_sum is None:
            strat_sum = np.zeros(NUM_ACTIONS, dtype=np.float32)
            self.strategy_sum[info] = strat_sum

        if active == traverser:
            util = np.zeros(NUM_ACTIONS, dtype=np.float32)
            node_util = 0.0
            legal_indices = np.nonzero(legal_mask)[0]
            for a in legal_indices:
                next_state = state.proceed(int(a))
                util[a] = self.traverse(next_state, traverser, reach_trav * strat[a], reach_opp)
                node_util += strat[a] * util[a]
            for a in legal_indices:
                # Under external sampling the opponent's sampling already
                # supplies the pi_{-i} weighting, so the correct update is the
                # unweighted (util[a] - node_util); multiplying by reach_opp
                # applies it twice and starves river nodes (~0.01 by street 5).
                # MCCFR_EXTERNAL_SAMPLING=1 opts in; the default keeps the old
                # behaviour so existing runs stay comparable.
                delta_r = util[a] - node_util
                regrets[a] += delta_r if _EXTERNAL_SAMPLING else reach_opp * delta_r
            strat_sum += reach_trav * strat
            return float(node_util)

        # Opponent node: let the pool bot choose if one is attached, else
        # sample from our own strategy (self-play).
        opp = getattr(self, '_pool_opponent', None)
        if opp is not None:
            a = opp.action_index(state, legal_mask)
            if a is not None:
                # No strat_sum here -- this action is the pool bot's, not our
                # policy's -- but reach_opp must still decay as it does in the
                # self-play branch, or pool traversals carry orders of magnitude
                # more regret mass and a 50/50 mix trains almost entirely on
                # them. strat[a] is our model's probability of the bot's action,
                # not the bot's own (black boxes, unaffordable to sample), but
                # it puts both traversal types on the same scale.
                next_state = state.proceed(int(a))
                return self.traverse(next_state, traverser, reach_trav,
                                     reach_opp * max(float(strat[a]),
                                                     MIN_OPP_REACH))
            # fall through to self-play if the action was not representable

        strat_sum += reach_opp * strat
        epsilon = self._epsilon(state.street)
        a = sample_action(strat, legal_mask, epsilon=epsilon)
        next_state = state.proceed(int(a))
        return self.traverse(next_state, traverser, reach_trav,
                             reach_opp * max(float(strat[a]), MIN_OPP_REACH))

    def average_strategy(self):
        avg = {}
        for info, strat_sum in self.strategy_sum.items():
            total = float(strat_sum.sum())
            if total <= 1e-12:
                avg[info] = (strat_sum + 0.0).tolist()
            else:
                avg[info] = (strat_sum / total).tolist()
        return avg

    def save_checkpoint(self, path):
        payload = {
            "version": 2,
            "regrets": self.regrets,
            "strategy_sum": self.strategy_sum,
            "score_ranges": self.score_ranges,
            "hand_buckets": self.hand_buckets,
            "board_buckets": self.board_buckets,
            "traversals": self.traversals,
            "pot_buckets": self.pot_buckets,
            "spr_buckets": self.spr_buckets,
            "pressure_buckets": self.pressure_buckets,
            "raise_fractions": dict(MCCFR_RAISE_FRACTIONS),
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_postflop_mult": self.epsilon_postflop_mult,
        }
        # Temp file then os.replace, which is atomic on POSIX and Windows: a
        # scancel or timeout mid-write leaves the previous checkpoint intact
        # rather than a truncated, unreadable one.
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def load_checkpoint(self, path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.regrets = payload.get("regrets", {})
        self.strategy_sum = payload.get("strategy_sum", {})
        self.score_ranges = payload.get("score_ranges", self.score_ranges)
        self.hand_buckets = payload.get("hand_buckets", self.hand_buckets)
        self.board_buckets = payload.get("board_buckets", self.board_buckets)
        self.traversals = int(payload.get("traversals", 0))
        self.pot_buckets = payload.get("pot_buckets", self.pot_buckets)
        self.spr_buckets = payload.get("spr_buckets", self.spr_buckets)
        self.pressure_buckets = payload.get("pressure_buckets", self.pressure_buckets)
        self.epsilon_start = float(payload.get("epsilon_start", self.epsilon_start))
        self.epsilon_end = float(payload.get("epsilon_end", self.epsilon_end))
        self.epsilon_decay = float(payload.get("epsilon_decay", self.epsilon_decay))
        self.epsilon_postflop_mult = float(payload.get("epsilon_postflop_mult", self.epsilon_postflop_mult))


def _mccfr_bool_env(name, default=False):
    """Boolean env var; unset -> default. Not bool(raw), which reads '0' as True."""
    raw = (os.environ.get(name) or '').strip().lower()
    if not raw:
        return default
    return raw not in ('0', 'false', 'no', 'off')


_EXTERNAL_SAMPLING = _mccfr_bool_env('MCCFR_EXTERNAL_SAMPLING')


def sample_action(strategy, legal_mask, epsilon=0.0):
    weights = strategy * legal_mask
    if epsilon > 1e-9:
        legal = legal_mask.sum()
        if legal > 0:
            uniform = legal_mask / legal
            weights = (1.0 - epsilon) * weights + epsilon * uniform
    total = float(weights.sum())
    if total <= 1e-12:
        legal = np.nonzero(legal_mask)[0]
        return int(random.choice(legal)) if len(legal) else CALL_CHECK
    r = random.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += float(w)
        if r <= acc:
            return i
    return int(np.nonzero(legal_mask)[0][-1])


def new_hand_state():
    deck = pkrbot.Deck()
    deck.shuffle()
    hands = [deck.deal(3), deck.deal(3)]
    board = []
    pips = [SMALL_BLIND, BIG_BLIND]
    stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
    return TossState(0, 0, pips, stacks, hands, board, deck, 0, -1, 0)


def sample_score_ranges(samples, buckets=None):
    """Quantile cut points for showdown scores, per board length.

    Quantiles, not (min, max): scores skew hard toward weak hands, so linear
    interpolation put ~70% of them in the bottom two buckets of eight.
    """
    ranks = "23456789TJQKA"
    suits = "cdhs"
    if buckets is None:
        buckets = {}
    ranges = {}
    for board_len in (3, 4, 5, 6):
        nb = int(select_bucket_count(buckets, board_len, 8)) if buckets else 8
        nb = max(2, nb)
        scores = []
        deck = [r + s for r in ranks for s in suits]
        for _ in range(samples):
            random.shuffle(deck)
            board = [pkrbot.Card(c) for c in deck[:board_len]]
            hand = [pkrbot.Card(c) for c in deck[board_len:board_len + 2]]
            scores.append(int(pkrbot.evaluate(board + hand)))
        scores.sort()
        if len(scores) < nb:
            ranges[board_len] = (scores[0], scores[-1]) if scores else (0, 1)
            continue
        cuts = [scores[int(round(i * len(scores) / nb)) - 1 if i else 0]
                for i in range(1, nb)]
        ranges[board_len] = cuts
    return ranges


def save_policy(path, trainer):
    payload = {
        "version": 2,
        "actions": list(MCCFR_ACTIONS),
        "raise_fractions": dict(MCCFR_RAISE_FRACTIONS),
        "pot_buckets": trainer.pot_buckets,
        "spr_buckets": trainer.spr_buckets,
        "pressure_buckets": trainer.pressure_buckets,
        "hand_buckets": trainer.hand_buckets,
        "board_buckets": trainer.board_buckets,
        "score_ranges": dict(trainer.score_ranges),
        "policy": trainer.average_strategy(),
        "traversals": int(trainer.traversals),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traversals", type=int, default=600000)
    parser.add_argument("--output", type=str, default=os.path.join(ROOT, "mccfr_bot", "mccfr_policy.pkl"))
    parser.add_argument("--seed", type=int, default=7)
    # Sharding: the score-range cut points decide what every info key means, so
    # shards must share them or merging sums unrelated states. Hold --range-seed
    # fixed across shards and vary --seed so they explore independently.
    parser.add_argument("--range-seed", type=int, default=None,
                        help="seed for score-range sampling only; hold fixed "
                             "across parallel shards so their tables can be merged")
    parser.add_argument("--hand-buckets", type=int, default=HAND_BUCKETS)
    parser.add_argument("--board-buckets", type=int, default=BOARD_BUCKETS)
    parser.add_argument("--range-samples", type=int, default=4000)
    parser.add_argument("--checkpoint", type=str, default=os.path.join(ROOT, "mccfr_bot", "mccfr_ckpt.pkl"))
    parser.add_argument("--checkpoint-every", type=int, default=50000)
    parser.add_argument("--epsilon-start", type=float, default=0.14)
    parser.add_argument("--epsilon-end", type=float, default=0.015)
    parser.add_argument("--epsilon-decay", type=float, default=1000000)
    parser.add_argument("--epsilon-postflop-mult", type=float, default=1.35)
    # Pure self-play converges to an abstract-game equilibrium that
    # BluffHeavyBot beats for ~20k chips, and more traversals make it worse.
    parser.add_argument("--pool-opponents", type=str, default="",
                        help="comma-separated opponent_pool bot names to train "
                             "against, optionally weighted as Name:weight, e.g. "
                             "'GoodBot:3,BluffHeavyBot:1,ManiacBot:1'. Weights "
                             "control how often each is drawn; unweighted names "
                             "default to 1. Empty = pure self-play.")
    parser.add_argument("--pool-prob", type=float, default=0.5,
                        help="fraction of traversals played against a pool bot "
                             "rather than self-play (only used when "
                             "--pool-opponents is set)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("[mccfr] Sampling score ranges...")
    # Cut points must match the bucket count the trainer actually uses.
    if args.range_seed is None:
        score_ranges = sample_score_ranges(args.range_samples, args.hand_buckets or HAND_BUCKETS)
    else:
        # Shard-independent stream for the cut points, then restore the
        # per-shard streams so traversals still differ.
        py_state, np_state = random.getstate(), np.random.get_state()
        random.seed(args.range_seed)
        np.random.seed(args.range_seed)
        score_ranges = sample_score_ranges(args.range_samples, args.hand_buckets or HAND_BUCKETS)
        random.setstate(py_state)
        np.random.set_state(np_state)
        print(f"[mccfr] score ranges drawn from --range-seed {args.range_seed} "
              f"(shardable); traversals use --seed {args.seed}")
    trainer = MCCFRTrainer(
        score_ranges,
        hand_buckets=args.hand_buckets,
        board_buckets=args.board_buckets,
        pot_buckets=POT_BB_BUCKETS,
        spr_buckets=SPR_BUCKETS,
        pressure_buckets=PRESSURE_BUCKETS,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        epsilon_postflop_mult=args.epsilon_postflop_mult,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[mccfr] Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # "Name" or "Name:weight". GoodBot is the only pool member that beats the
    # others, so weighting it up puts traversals on the hard decisions.
    pool_adapters = []
    pool_weights = []
    for tok in (args.pool_opponents or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            nm, _, w = tok.partition(":")
            try:
                weight = max(0.0, float(w))
            except ValueError:
                raise SystemExit(f"ERROR: bad weight in --pool-opponents entry {tok!r}")
        else:
            nm, weight = tok, 1.0
        if weight <= 0:
            continue
        pool_adapters.append(PoolOpponentAdapter(nm.strip(), seed=args.seed))
        pool_weights.append(weight)

    pool_cum = []
    if pool_adapters:
        total_w = sum(pool_weights)
        acc = 0.0
        for w in pool_weights:
            acc += w
            pool_cum.append(acc / total_w)
        shares = ", ".join(f"{a.name} {100*w/total_w:.0f}%"
                           for a, w in zip(pool_adapters, pool_weights))
        print(f"[mccfr] training vs pool at pool-prob {args.pool_prob:.2f} "
              f"(rest self-play): {shares}")
    else:
        print("[mccfr] pure self-play (no --pool-opponents given)")

    start = time.time()
    for t in range(args.traversals):
        # Both players' traversals share an opponent so the hand is coherent.
        opp = None
        if pool_adapters and random.random() < float(args.pool_prob):
            r = random.random()
            idx = next((i for i, c in enumerate(pool_cum) if r <= c), len(pool_adapters) - 1)
            opp = pool_adapters[idx]
        trainer._pool_opponent = opp
        for player in (0, 1):
            # new_hand_state() deals a fresh hand per traversal, so the bot
            # needs a new-round signal each time or state latches across hands
            # (UltraOptimizedBot.kept_cards ends up naming absent cards).
            if opp is not None:
                opp.new_hand(trainer.traversals + 1 + player)
            state = new_hand_state()
            trainer.traverse(state, player, 1.0, 1.0)
        trainer._pool_opponent = None
        trainer.traversals += 2
        if args.checkpoint and args.checkpoint_every > 0 and trainer.traversals % args.checkpoint_every == 0:
            trainer.save_checkpoint(args.checkpoint)
        if (t + 1) % 5000 == 0:
            elapsed = time.time() - start
            print(f"[mccfr] traversals={trainer.traversals} infosets={len(trainer.regrets)} elapsed={elapsed:.1f}s")

    save_policy(args.output, trainer)
    print(f"[mccfr] Wrote policy to {args.output}")


if __name__ == "__main__":
    main()
