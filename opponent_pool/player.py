'''
Picks one opponent style per engine run and uses it for every hand.

Env: POOL_SEED (reproducible pick), POOL_OPPONENT (force a style, e.g. GoodBot
or SelfPlaySnapshot), SELFPLAY_PROB (chance of frozen self-play per match).
'''

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
import importlib.util
from pathlib import Path
from typing import Union

_RANK_CHARS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
_RANK_TO_INT = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

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

_GOODBOT_PARAMS_CACHE = {}

def _goodbot_params() -> dict:
    path = (os.environ.get('GOODBOT_PARAMS_PATH') or '').strip()
    if not path:
        return {}
    if path in _GOODBOT_PARAMS_CACHE:
        return _GOODBOT_PARAMS_CACHE[path]
    vals = {}
    try:
        import json as _json
        with open(path, 'r', encoding='utf-8') as f:
            raw = _json.load(f)
        vals = {k: float(v) for k, v in raw.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
        print(f"[params] loaded {len(vals)} tuned knobs from {path}")
    except Exception as e:
        print(f"[params] could not load {path}: {e}")
        vals = {}
    _GOODBOT_PARAMS_CACHE[path] = vals
    return vals

def _p(name: str, default: float) -> float:
    try:
        return float(_goodbot_params().get(name, default))
    except Exception:
        return default

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

class UltraTightBot:
    
    def __init__(self):
        self.hand = None
        self.kept_cards = None
    
    def handle_new_round(self, game_state, round_state, active):
        self.hand = None
        self.kept_cards = None
    
    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        my_cards = round_state.hands[active]
        continue_cost = round_state.pips[1 - active] - round_state.pips[active]
        
        if DiscardAction in legal_actions:
            ranks = [_RANK_TO_INT[_card_rank_char(c)] for c in my_cards]
            min_index = ranks.index(min(ranks))
            return DiscardAction(min_index)

        if self.kept_cards is None:
            self.kept_cards = my_cards
        
        def _strength2(c1, c2) -> float:
            r1 = _card_rank_char(c1)
            r2 = _card_rank_char(c2)
            ranks = [r1, r2]
            suited = _are_suited(c1, c2)

            if r1 == r2:
                if r1 in ['A', 'K', 'Q', 'J']:
                    return 0.9
                if r1 in ['T', '9']:
                    return 0.6
                return 0.0

            strength = 0.0
            if 'A' in ranks and ('K' in ranks or 'Q' in ranks):
                strength = 0.8
            if suited and set(ranks) <= {'A', 'K', 'Q', 'J'}:
                strength = max(strength, 0.7)
            return strength

        hand_strength = 0.0
        for c1, c2 in _two_card_combos(self.kept_cards):
            if len((c1, c2)) == 2:
                hand_strength = max(hand_strength, _strength2(c1, c2))

        if continue_cost > 0:
            return CallAction() if hand_strength > 0.7 and CallAction in legal_actions else FoldAction()

        if hand_strength > 0.7 and RaiseAction in legal_actions:
            return _safe_raise(round_state, active, 0.7)
        return CheckAction() if CheckAction in legal_actions else (CallAction() if CallAction in legal_actions else FoldAction())

class UltraLooseBot:
    
    def __init__(self):
        self.kept_cards = None
    
    def handle_new_round(self, game_state, round_state, active):
        self.kept_cards = None
    
    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        continue_cost = round_state.pips[1 - active] - round_state.pips[active]
        
        if DiscardAction in legal_actions:
            return DiscardAction(random.randint(0, 2))
        
        if self.kept_cards is None:
            self.kept_cards = round_state.hands[active]

        if continue_cost > 0:
            rand = random.random()
            if rand < 0.7 and CallAction in legal_actions:
                return CallAction()
            if rand < 0.9 and RaiseAction in legal_actions:
                return _safe_raise(round_state, active, 0.3)
            return FoldAction()

        if random.random() < 0.6 and RaiseAction in legal_actions:
            return _safe_raise(round_state, active, 0.4)
        return CheckAction() if CheckAction in legal_actions else (CallAction() if CallAction in legal_actions else FoldAction())

class CallingStationBot:
    
    def __init__(self):
        self.kept_cards = None
    
    def handle_new_round(self, game_state, round_state, active):
        self.kept_cards = None
    
    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        continue_cost = round_state.pips[1 - active] - round_state.pips[active]
        
        if DiscardAction in legal_actions:
            return DiscardAction(random.randint(0, 2))
        
        if self.kept_cards is None:
            self.kept_cards = round_state.hands[active]

        if continue_cost > 0:
            return CallAction() if random.random() < 0.9 and CallAction in legal_actions else FoldAction()

        if random.random() < 0.1 and RaiseAction in legal_actions:
            min_raise, _ = round_state.raise_bounds()
            return RaiseAction(min_raise)
        return CheckAction() if CheckAction in legal_actions else (CallAction() if CallAction in legal_actions else FoldAction())

class ManiacBot:
    
    def __init__(self):
        self.kept_cards = None
    
    def handle_new_round(self, game_state, round_state, active):
        self.kept_cards = None
    
    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        continue_cost = round_state.pips[1 - active] - round_state.pips[active]
        
        if DiscardAction in legal_actions:
            my_cards = round_state.hands[active]
            ranks = [_RANK_TO_INT[_card_rank_char(c)] for c in my_cards]
            min_index = ranks.index(min(ranks))
            return DiscardAction(min_index)
        
        if self.kept_cards is None:
            self.kept_cards = round_state.hands[active]

        if continue_cost > 0:
            rand = random.random()
            if rand < 0.7 and RaiseAction in legal_actions:
                _, max_raise = round_state.raise_bounds()
                return RaiseAction(max_raise)
            if rand < 0.9 and CallAction in legal_actions:
                return CallAction()
            return FoldAction()

        if RaiseAction in legal_actions:
            _, max_raise = round_state.raise_bounds()
            return RaiseAction(max_raise)
        return CheckAction() if CheckAction in legal_actions else (CallAction() if CallAction in legal_actions else FoldAction())

class RandomBot:
    
    def __init__(self):
        self.kept_cards = None
    
    def handle_new_round(self, game_state, round_state, active):
        self.kept_cards = None
    
    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal_actions = list(round_state.legal_actions())
        
        if DiscardAction in legal_actions:
            return DiscardAction(random.randint(0, 2))
        
        if self.kept_cards is None:
            self.kept_cards = round_state.hands[active]
        
        legal_actions = [a for a in legal_actions if a != DiscardAction]
        
        if not legal_actions:
            return FoldAction()
            
        action_class = random.choice(legal_actions)

        if action_class == RaiseAction:
            min_raise, max_raise = round_state.raise_bounds()
            amount = random.randint(min_raise, max_raise)
            return RaiseAction(amount)

        return action_class()

class BluffHeavyBot:
    
    def __init__(self):
        self.kept_cards = None
    
    def handle_new_round(self, game_state, round_state, active):
        self.kept_cards = None
    
    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        street = round_state.street
        continue_cost = round_state.pips[1 - active] - round_state.pips[active]
        
        if DiscardAction in legal_actions:
            return DiscardAction(random.randint(0, 2))
        
        if self.kept_cards is None:
            self.kept_cards = round_state.hands[active]

        bluff_probability = 0.3 if street == 0 else 0.6

        if continue_cost > 0:
            if random.random() < bluff_probability and RaiseAction in legal_actions:
                return _safe_raise(round_state, active, 0.8)
            if random.random() < 0.7 and CallAction in legal_actions:
                return CallAction()
            return FoldAction()

        if random.random() < bluff_probability and RaiseAction in legal_actions:
            return _safe_raise(round_state, active, 0.5)
        return CheckAction() if CheckAction in legal_actions else (CallAction() if CallAction in legal_actions else FoldAction())

class RockBot:
    
    def __init__(self):
        self.kept_cards = None
    
    def handle_new_round(self, game_state, round_state, active):
        self.kept_cards = None
    
    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        my_cards = round_state.hands[active]
        continue_cost = round_state.pips[1 - active] - round_state.pips[active]
        
        if DiscardAction in legal_actions:
            ranks = [_RANK_TO_INT[_card_rank_char(c)] for c in my_cards]
            min_index = ranks.index(min(ranks))
            return DiscardAction(min_index)
        
        if self.kept_cards is None:
            self.kept_cards = my_cards
        
        def _strength2(c1, c2) -> float:
            ranks = [_card_rank_char(c1), _card_rank_char(c2)]
            if ranks[0] == ranks[1] and ranks[0] in ['A', 'K', 'Q']:
                return 0.9
            if set(ranks) == {'A', 'K'}:
                return 0.8
            return 0.0

        hand_strength = 0.0
        for c1, c2 in _two_card_combos(self.kept_cards):
            if len((c1, c2)) == 2:
                hand_strength = max(hand_strength, _strength2(c1, c2))

        if continue_cost > 0:
            return CallAction() if hand_strength > 0.8 and CallAction in legal_actions else FoldAction()

        if hand_strength > 0.9 and RaiseAction in legal_actions:
            min_raise, _ = round_state.raise_bounds()
            return RaiseAction(min_raise)
        return CheckAction() if CheckAction in legal_actions else (CallAction() if CallAction in legal_actions else FoldAction())

class BalancedBot:
    
    def __init__(self):
        self.kept_cards = None
    
    def handle_new_round(self, game_state, round_state, active):
        self.kept_cards = None
    
    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def calculate_hand_strength(self, my_cards) -> float:
        def _calc2(c1, c2) -> float:
            ranks = sorted([_RANK_TO_INT[_card_rank_char(c1)], _RANK_TO_INT[_card_rank_char(c2)]], reverse=True)
            pair = (ranks[0] == ranks[1])
            suited = _are_suited(c1, c2)

            if pair:
                return min(0.9, 0.25 + ranks[0] / 20.0)

            high_cards = sum(1 for r in ranks if r >= 11)
            strength = 0.25 + 0.15 * high_cards + (0.08 if suited else 0.0)
            return max(0.0, min(1.0, strength))

        best = 0.0
        for c1, c2 in _two_card_combos(my_cards):
            if len((c1, c2)) == 2:
                best = max(best, _calc2(c1, c2))
        return best

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        my_cards = round_state.hands[active]
        continue_cost = round_state.pips[1 - active] - round_state.pips[active]
        
        if DiscardAction in legal_actions:
            best_combo = None
            best_score = -1
            
            combos = [(0, 1), (0, 2), (1, 2)]
            
            for i, j in combos:
                combo_cards = [my_cards[i], my_cards[j]]
                score = self.calculate_hand_strength(combo_cards)
                if score > best_score:
                    best_score = score
                    best_combo = (i, j)
            
            discard_idx = 3 - sum(best_combo)
            return DiscardAction(discard_idx)
        
        if self.kept_cards is None:
            self.kept_cards = my_cards
        
        hand_strength = self.calculate_hand_strength(self.kept_cards)

        if continue_cost > 0:
            pot_so_far = (STARTING_STACK - round_state.stacks[active]) + (STARTING_STACK - round_state.stacks[1 - active])
            pot_odds = continue_cost / max(1.0, continue_cost + pot_so_far)
            required_equity = min(1.0, pot_odds)

            if hand_strength > required_equity * 0.9:
                if RaiseAction in legal_actions and hand_strength > 0.55 and random.random() < 0.45:
                    return _safe_raise(round_state, active, 0.45)
                return CallAction() if CallAction in legal_actions else CheckAction()
            return FoldAction()

        if RaiseAction in legal_actions:
            if hand_strength > 0.5 and random.random() < 0.8:
                return _safe_raise(round_state, active, max(0.35, hand_strength))
            if hand_strength < 0.35 and random.random() < 0.25:
                min_raise, _ = round_state.raise_bounds()
                return RaiseAction(min_raise)

        return CheckAction() if CheckAction in legal_actions else (CallAction() if CallAction in legal_actions else FoldAction())

class GoodBot:
    
    def __init__(self):
        self.kept_cards = None
        self.hand_strength_cache = {}
        self._equity_mc_cache = {}
        self._opp_sampler_cache = {}
    
    def handle_new_round(self, game_state, round_state, active):
        self.kept_cards = None
    
    def handle_round_over(self, game_state, terminal_state, active):
        pass
    
    def calculate_hand_potential(self, cards, board_cards, iters=100):
        try:
            cards_key = tuple(sorted(str(c) for c in cards))
            board_key = tuple(sorted(str(c) for c in (board_cards or [])))
            cache_key = ('potential', cards_key, board_key, int(iters))
            if cache_key in self.hand_strength_cache:
                return self.hand_strength_cache[cache_key]
        except Exception:
            cache_key = None

        if len(cards) == 3:
            best = 0.0
            for i in range(3):
                for j in range(i + 1, 3):
                    kept = [cards[i], cards[j]]
                    best = max(best, self.calculate_equity(kept, [], iters=int(iters), opp_bias=0.0))
            result = best
            if cache_key is not None:
                _cache_put(self.hand_strength_cache, cache_key, result, 65536)
            return result
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
    
    def calculate_equity(self, my_cards, board_cards, iters=100, opp_bias=0.0):
        try:
            my_key = tuple(sorted(str(c) for c in my_cards))
            board_key = tuple(sorted(str(c) for c in (board_cards or [])))
            bias_bucket = round(max(0.0, min(1.0, float(opp_bias))) * 10.0) / 10.0
            state_key = (my_key, board_key, bias_bucket)
        except Exception:
            state_key = None

        remaining = 6 - len(board_cards)
        
        my_hand = [pkrbot.Card(str(card)) for card in my_cards]
        board_hand = [pkrbot.Card(str(card)) for card in board_cards]

        target_iters = int(iters)
        if target_iters <= 0:
            return 0.5

        bias = 0.0
        try:
            bias = max(0.0, min(1.0, float(opp_bias)))
        except Exception:
            bias = 0.0

        def _hand_strength_proxy(c1: str, c2: str) -> float:
            r1 = _RANK_TO_INT.get(_card_rank_char(c1), 2)
            r2 = _RANK_TO_INT.get(_card_rank_char(c2), 2)
            hi = max(r1, r2)
            lo = min(r1, r2)
            pair = (r1 == r2)
            suited = _are_suited(c1, c2)
            gap = abs(r1 - r2)

            if pair:
                s = 0.55 + (hi - 2) / 12.0 * 0.45
            else:
                s = 0.15 + (hi - 2) / 12.0 * 0.45 + (lo - 2) / 12.0 * 0.25
                if suited:
                    s += 0.08
                if gap <= 1:
                    s += 0.06
                elif gap <= 3:
                    s += 0.03

            return max(0.0, min(1.0, s))

        def _get_opp_sampler():
            if state_key is None:
                return None
            if state_key in self._opp_sampler_cache:
                return self._opp_sampler_cache[state_key]

            deck = pkrbot.Deck()
            for card in my_hand + board_hand:
                try:
                    deck.cards.remove(card)
                except ValueError:
                    pass
            remaining_cards = [str(c) for c in deck.cards]

            combos = []
            cum = []
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

            sampler = (combos, cum, total)
            _cache_put(self._opp_sampler_cache, state_key, sampler, 512)
            return sampler

        opp_sampler = _get_opp_sampler()

        def _sample_opp_hand_strings():
            if not opp_sampler:
                deck = pkrbot.Deck()
                for card in my_hand + board_hand:
                    try:
                        deck.cards.remove(card)
                    except ValueError:
                        pass
                deck.shuffle()
                h = deck.deal(2)
                return (str(h[0]), str(h[1]))

            combos, cum, total = opp_sampler
            if not combos or total <= 0.0:
                deck = pkrbot.Deck()
                for card in my_hand + board_hand:
                    try:
                        deck.cards.remove(card)
                    except ValueError:
                        pass
                deck.shuffle()
                h = deck.deal(2)
                return (str(h[0]), str(h[1]))

            r = random.random() * total
            idx = bisect.bisect_left(cum, r)
            if idx >= len(combos):
                idx = len(combos) - 1
            return combos[idx]

        def _one_rollout() -> float:
            deck = pkrbot.Deck()
            for card in my_hand + board_hand:
                try:
                    deck.cards.remove(card)
                except ValueError:
                    pass

            o1, o2 = _sample_opp_hand_strings()
            opp_hand = [pkrbot.Card(o1), pkrbot.Card(o2)]
            for card in opp_hand:
                try:
                    deck.cards.remove(card)
                except ValueError:
                    pass

            deck.shuffle()
            remaining_board = deck.deal(remaining)
            full_board = board_hand + remaining_board

            my_strength = pkrbot.evaluate(my_hand + full_board)
            opp_strength = pkrbot.evaluate(opp_hand + full_board)
            if my_strength > opp_strength:
                return 1.0
            if my_strength == opp_strength:
                return 0.5
            return 0.0

        if state_key is not None and state_key in self._equity_mc_cache:
            entry = self._equity_mc_cache[state_key]
        else:
            entry = {'samples': [], 'sum': 0.0}
            if state_key is not None:
                _cache_put(self._equity_mc_cache, state_key, entry, 4096)

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

        refresh = max(1, min(5, target_iters // 50))
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

        opp_bias = 0.0
        if continue_cost > 0:
            denom = float(max(1, pot + continue_cost))
            pressure = float(continue_cost) / denom
            opp_bias = max(0.0, min(1.0, 2.0 * pressure))

        rounds_left = max(0, int(NUM_ROUNDS - game_state.round_num) + 1)
        if float(game_state.bankroll) > 1.5 * float(rounds_left)+1:
            if DiscardAction not in legal_actions:
                if continue_cost > 0:
                    return FoldAction()
                return CheckAction() if CheckAction in legal_actions else FoldAction()
        
        if DiscardAction in legal_actions:
            iters = 150
            iters_per = max(1, iters // 3)

            best_discard = 0
            best_equity = -1.0

            for discard_idx in (0, 1, 2):
                kept = [my_cards[i] for i in (0, 1, 2) if i != discard_idx]
                board_after = list(board_cards) + [my_cards[discard_idx]]
                equity = self.calculate_equity(kept, board_after, iters=iters_per)

                if equity > best_equity:
                    best_equity = equity
                    best_discard = discard_idx

            return DiscardAction(best_discard)
        
        if self.kept_cards is None and len(my_cards) == 2:
            self.kept_cards = my_cards
        
        iters = 300 if street == 0 else 200 if street == 3 else 150
        
        equity = 0.5
        if self.kept_cards is not None and len(board_cards) >= 4:
            equity = self.calculate_equity(self.kept_cards, board_cards, iters=iters, opp_bias=opp_bias)
        elif street == 0:
            equity = self.calculate_hand_potential(my_cards, [], iters=min(200, int(iters)))
        
        if continue_cost > 0:
            if continue_cost > my_stack:
                continue_cost = my_stack
            
            pot_after_call = my_contribution + opp_contribution + continue_cost
            pot_odds = continue_cost / (pot_after_call - my_contribution) if (pot_after_call - my_contribution) > 0 else 1.0
            required_equity = pot_odds * _p('call_equity_multiplier', 1.0)
            
            if equity > required_equity:
                opp_equity_approx = 1 - equity
                if RaiseAction in legal_actions:
                    min_raise, max_raise = round_state.raise_bounds()
                    _hi = _p('facing_bet_raise_pot_hi', 1.0)
                    _lo = _p('facing_bet_raise_pot_lo', 0.5)
                    if equity >= _p('raise_size_equity_hi', 0.62):
                        size = _hi
                    elif equity >= _p('raise_size_equity_mid', 0.54):
                        size = (_lo + _hi) / 2.0
                    else:
                        size = _lo
                    raise_amount = int(my_pip + continue_cost + pot_after_call * size)
                    raise_amount = min(max_raise, max(min_raise, raise_amount))
                    raise_cost = raise_amount - my_pip - continue_cost
                    p_fold = (_p('p_fold_base', 0.0)
                              + _p('p_fold_equity_mult', 0.4) * (1 - opp_equity_approx))
                    p_fold = min(1.0, max(0.0, p_fold))
                    ev_if_fold = opp_contribution
                    ev_if_call = equity * (opp_contribution + (raise_amount - opp_pip)) - (1-equity) * (my_contribution + continue_cost + raise_cost)
                    ev_raise = p_fold * ev_if_fold + (1 - p_fold) * ev_if_call
                    ev_call = equity * pot_after_call - (1-equity) * (my_contribution + continue_cost)
                    
                    if (ev_raise > ev_call + _p('raise_if_ev_margin', 0.0)
                            or (equity < _p('raise_bluff_max_equity', 0.4)
                                and random.random() < _p('raise_bluff_prob', 0.2))):
                        return RaiseAction(raise_amount)
                return CallAction()
            else:
                return FoldAction()
        else:
            if RaiseAction in legal_actions and (
                    equity > _p('bet_value_threshold', 0.65)
                    or (equity < _p('bet_bluff_max_equity', 0.45)
                        and random.random() < _p('bet_bluff_prob', 0.15))):
                min_raise, max_raise = round_state.raise_bounds()
                size = (_p('unchecked_bet_pot_hi', 0.75) if equity > _p('bet_value_threshold', 0.65)
                        else _p('unchecked_bet_pot_lo', 0.5))
                raise_amount = int(my_pip + pot * size)
                raise_amount = min(max_raise, max(min_raise, raise_amount))
                return RaiseAction(raise_amount)
            return CheckAction()

class UltraOptimizedBot:
    def __init__(self):
        self.params = {
            "aggression": 0.11495355077976095,
            "tightness": 0,
            "bluff_frequency": 0,
            "use_ev_decision": 0.8532086683470317,
            "position_bonus": 5,
            "value_bet_size": 0,
            "bluff_bet_size": 0,
            "raise_size": 2.4677254856087676,
            "vpip_threshold": 0.3472184082406298,
            "pfr_threshold": 0.609805764164761,
            "min_equity_for_bluff": 0.07731604926093129,
            "max_equity_for_value": 1,
            "call_threshold_mult": 0.6261226964184712,
            "pot_odds_margin": 1.716753549127868,
            "position_aggression_mult": 5,
            "position_ev_multiplier": 5,
            "preflop_aggression_mult": 2.0747305180469606,
            "flop_aggression_mult": 5,
            "turn_aggression_mult": 1.727059708989133,
            "river_aggression_mult": 0,
            "winning_aggression_boost": 4.690374478084194,
            "losing_tightness_boost": 0,
            "stack_size_adjustment": 0.31698392608369697,
            "positive_ev_aggression_boost": 2.1144187757297783,
            "preflop_iterations": 300,
            "postflop_iterations": 200,
            "wet_board_aggression_mult": 0.01632022142669405,
            "dry_board_aggression_mult": 5,
            "opponent_exploitation_weight": 3.1583964643346554,
            "fold_equity_adjustment": 4.664234848887343,
            "gto_blend_weight": 0.5636724213585388,
            "balanced_range_weight": 0,
            "adaptive_iterations_weight": 0.009294831339310297
        }
        self.equity_cache = {}
        self.kept_cards = None

    def _coerce_pkrbot_cards(self, cards):
        if not cards:
            return []
        
        coerced = []
        for card in cards:
            if hasattr(card, 'rank') and hasattr(card, 'suit'):
                coerced.append(card)
            else:
                coerced.append(pkrbot.Card(str(card)))
        return coerced
    
    def handle_new_round(self, game_state, round_state, active):
        self.kept_cards = None

    def handle_round_over(self, game_state, terminal_state, active):
        pass
            
    def calculate_equity(self, my_cards, board_cards, iters=200):
        my_cards_str = [str(card) for card in my_cards]
        board_cards_str = [str(card) for card in board_cards]
        
        cache_key = (tuple(sorted(my_cards_str)), tuple(sorted(board_cards_str)), iters)
        
        if cache_key in self.equity_cache:
            return self.equity_cache[cache_key]
        
        remaining = 6 - len(board_cards)
        my_hand = [pkrbot.Card(card) for card in my_cards_str]
        board_hand = [pkrbot.Card(card) for card in board_cards_str]

        wins, ties = 0, 0
        for _ in range(iters):
            deck = pkrbot.Deck()
            for card in my_hand + board_hand:
                try:
                    deck.cards.remove(card)
                except ValueError:
                    pass
            deck.shuffle()
            
            opp_hand = deck.deal(2)
            remaining_board = deck.deal(remaining)
            full_board = board_hand + remaining_board
            
            my_strength = pkrbot.evaluate(my_hand + full_board)
            opp_strength = pkrbot.evaluate(opp_hand + full_board)
            
            if my_strength > opp_strength:
                wins += 1
            elif my_strength == opp_strength:
                ties += 1
        
        result = (wins + ties * 0.5) / iters
        self.equity_cache[cache_key] = result
        return result
    
    def calculate_preflop_potential(self, cards):
        if len(cards) != 3:
            return 0.5
        
        ranks = [_RANK_TO_INT[_card_rank_char(c)] for c in cards]
        suits = [str(c.suit) if hasattr(c, 'suit') else str(c)[1] for c in cards]
        
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        
        max_count = max(rank_counts.values())
        
        if max_count >= 2:
            pair_rank = [r for r, c in rank_counts.items() if c >= 2][0]
            base = 0.3 + pair_rank / 26.0
            
            other_ranks = [r for r in ranks if r != pair_rank]
            if other_ranks:
                base += max(other_ranks) / 52.0
            
            return min(0.9, base)
        
        sorted_ranks = sorted(ranks)
        suited = len(set(suits)) < 3
        
        gap = max(sorted_ranks) - min(sorted_ranks)
        
        if gap <= 4:
            base = 0.25 + max(sorted_ranks) / 52.0
            if suited:
                base += 0.1
            return min(0.8, base)
        
        high_cards = sum(1 for r in ranks if r >= 11)
        base = 0.2 + 0.1 * high_cards
        if suited:
            base += 0.05
        
        return min(0.7, base)
    
    def analyze_board_texture(self, board_cards):
        board_cards = self._coerce_pkrbot_cards(board_cards)
        if not board_cards:
            return {'wetness': 0.0, 'type': 'preflop'}
        
        card_rank_map = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 
                         7: '9', 8: 'T', 9: 'J', 10: 'Q', 11: 'K', 12: 'A'}
        
        ranks = [card.rank for card in board_cards]
        suits = [card.suit for card in board_cards]
        
        suit_counts = {}
        for suit in suits:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        max_suit = max(suit_counts.values(), default=0)
        flush_draw = max_suit >= 4 if len(board_cards) >= 4 else max_suit >= 3
        
        sorted_ranks = sorted(set(ranks))
        straight_draw = False
        if len(sorted_ranks) >= 3:
            for i in range(len(sorted_ranks) - 2):
                if sorted_ranks[i+2] - sorted_ranks[i] <= 4:
                    straight_draw = True
                    break
        
        paired = len(set(ranks)) < len(ranks)
        
        wetness = 0.0
        if flush_draw:
            wetness += 0.4
        if straight_draw:
            wetness += 0.4
        if paired:
            wetness -= 0.3
        
        if len(sorted_ranks) >= 3:
            gaps = 0
            for i in range(len(sorted_ranks) - 1):
                gap = sorted_ranks[i+1] - sorted_ranks[i]
                if gap == 1:
                    wetness += 0.1
                elif gap == 2:
                    wetness += 0.05
        
        wetness = max(0.0, min(wetness, 1.0))
        
        if wetness > 0.6:
            board_type = 'very_wet'
        elif wetness > 0.3:
            board_type = 'wet'
        elif wetness > 0.1:
            board_type = 'semi_wet'
        else:
            board_type = 'dry'
        
        return {'wetness': wetness, 'type': board_type, 'paired': paired, 
                'flush_draw': flush_draw, 'straight_draw': straight_draw}
    
    def get_action(self, game_state, round_state, player_index):
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = self._coerce_pkrbot_cards(round_state.hands[player_index])
        board_cards = self._coerce_pkrbot_cards(round_state.board)
        my_pip = round_state.pips[player_index]
        opp_pip = round_state.pips[1 - player_index]
        my_stack = round_state.stacks[player_index]
        opp_stack = round_state.stacks[1 - player_index]
        continue_cost = opp_pip - my_pip
        is_dealer = (player_index == round_state.button)
        my_contribution = STARTING_STACK - my_stack
        opp_contribution = STARTING_STACK - opp_stack
        current_pot = my_contribution + opp_contribution
        
        if DiscardAction in legal_actions:
            ranks = [_RANK_TO_INT[_card_rank_char(c)] for c in my_cards]
            min_index = ranks.index(min(ranks))
            return DiscardAction(min_index)
        
        if self.kept_cards is None and len(my_cards) == 2:
            self.kept_cards = my_cards
        
        if street == 0:
            cards_to_use = my_cards
            equity = self.calculate_preflop_potential(cards_to_use)
        elif self.kept_cards is not None:
            cards_to_use = self.kept_cards
            iters = 200 if street == 3 else 150
            equity = self.calculate_equity(cards_to_use, board_cards, iters)
        else:
            cards_to_use = my_cards
            equity = 0.5
        
        board_texture = self.analyze_board_texture(board_cards)
        
        base_aggression = self.params.get('aggression', 0.5)
        
        street_mult_key = {
            0: 'preflop_aggression_mult',
            3: 'flop_aggression_mult',
            4: 'turn_aggression_mult',
            5: 'river_aggression_mult'
        }.get(street, None)
        
        if street_mult_key:
            base_aggression *= self.params.get(street_mult_key, 1.0)
        
        if board_texture['wetness'] > 0.5:
            board_aggression_mult = self.params.get('wet_board_aggression_mult', 1.5)
            base_aggression *= board_aggression_mult
        elif board_texture['wetness'] < 0.2:
            board_aggression_mult = self.params.get('dry_board_aggression_mult', 0.8)
            base_aggression *= board_aggression_mult
        
        if is_dealer:
            base_aggression *= self.params.get('position_aggression_mult', 1.0)
        
        aggression = base_aggression
        
        if continue_cost > 0:
            if continue_cost > my_stack:
                continue_cost = my_stack
            
            pot_odds = continue_cost / (current_pot + continue_cost) if continue_cost > 0 else 0
            
            required_equity = pot_odds * self.params.get('call_threshold_mult', 1.0)
            if is_dealer:
                required_equity *= 0.9
            
            required_equity *= self.params.get('pot_odds_margin', 1.0)
            required_equity *= (1.0 + self.params.get('tightness', 0.0))
            
            use_ev_decision = self.params.get('use_ev_decision', 0.5)
            
            if random.random() < use_ev_decision:
                if equity > required_equity:
                    if RaiseAction in legal_actions and random.random() < aggression:
                        if random.random() < self.params.get('pfr_threshold', 1.0):
                            min_raise, max_raise = round_state.raise_bounds()
                            raise_amount = int(my_pip + continue_cost + current_pot * 
                                              self.params.get('raise_size', 0.5))
                            raise_amount = min(max_raise, max(min_raise, raise_amount))
                            return RaiseAction(raise_amount)
                    return CallAction()
                else:
                    return FoldAction()
            else:
                if random.random() > self.params.get('vpip_threshold', 1.0):
                    return FoldAction()
                
                if equity > required_equity:
                    if RaiseAction in legal_actions and random.random() < aggression:
                        if random.random() < self.params.get('pfr_threshold', 1.0):
                            min_raise, max_raise = round_state.raise_bounds()
                            raise_amount = int(my_pip + continue_cost + current_pot * 
                                              self.params.get('raise_size', 0.5))
                            raise_amount = min(max_raise, max(min_raise, raise_amount))
                            return RaiseAction(raise_amount)
                    return CallAction()
                else:
                    return FoldAction()
        else:
            should_bet = False
            
            value_threshold = self.params.get('max_equity_for_value', 0.7)
            if board_texture['wetness'] > 0.6:
                value_threshold *= 0.9
            
            if equity > value_threshold:
                should_bet = True
            
            elif equity < self.params.get('min_equity_for_bluff', 0.3):
                bluff_freq = self.params.get('bluff_frequency', 0.2)
                if random.random() < bluff_freq:
                    should_bet = True
            
            elif is_dealer and street > 0 and random.random() < aggression * 0.5:
                should_bet = True
            
            if should_bet and RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                
                if equity > value_threshold:
                    bet_size = int(current_pot * self.params.get('value_bet_size', 0.5))
                else:
                    bet_size = int(current_pot * self.params.get('bluff_bet_size', 0.3))
                
                bet_amount = my_pip + bet_size
                bet_amount = min(max_raise, max(min_raise, bet_amount))
                
                if bet_amount > my_pip:
                    return RaiseAction(bet_amount)
            
            return CheckAction()

class ReferenceBot:

    def __init__(self):
        pass

    def handle_new_round(self, game_state, round_state, active):
        my_bankroll = game_state.bankroll
        game_clock = game_state.game_clock
        round_num = game_state.round_num
        my_cards = round_state.hands[active]
        big_blind = bool(active)
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        my_delta = terminal_state.deltas[active]
        previous_state = terminal_state.previous_state
        street = previous_state.street
        my_cards = previous_state.hands[active]
        opp_cards = previous_state.hands[1-active]
        pass

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.board
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1-active]
        continue_cost = opp_pip - my_pip
        my_contribution = STARTING_STACK - my_stack
        opp_contribution = STARTING_STACK - opp_stack

        if DiscardAction in legal_actions:

            ranks = "23456789TJQKA"
            rank_list = [-1, -1, -1]

            for card in range(3):
                rank_list[card] = ranks.index(_card_rank_char(my_cards[card]))

            if rank_list[0] <= rank_list[1] and rank_list[0] <= rank_list[2]:
                return DiscardAction(0)
            elif rank_list[1] <= rank_list[2]:
                return DiscardAction(1)
            else:
                return DiscardAction(2)
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()
            min_cost = min_raise - my_pip
            max_cost = max_raise - my_pip
            if random.random() < 0.5:
                return RaiseAction(min_raise)
        if CheckAction in legal_actions:
            return CheckAction()
        if random.random() < 0.25:
            return FoldAction()
        return CallAction()

class TossGoodCards:

    def __init__(self):
        pass

    def handle_new_round(self, game_state, round_state, active):
        my_bankroll = game_state.bankroll
        game_clock = game_state.game_clock
        round_num = game_state.round_num
        my_cards = round_state.hands[active]
        big_blind = bool(active)
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        my_delta = terminal_state.deltas[active]
        previous_state = terminal_state.previous_state
        street = previous_state.street
        my_cards = previous_state.hands[active]
        opp_cards = previous_state.hands[1-active]
        pass

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.board
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1-active]
        continue_cost = opp_pip - my_pip
        my_contribution = STARTING_STACK - my_stack
        opp_contribution = STARTING_STACK - opp_stack

        if DiscardAction in legal_actions:

            ranks = "23456789TJQKA"
            rank_list = [-1, -1, -1]

            for card in range(3):
                rank_list[card] = ranks.index(_card_rank_char(my_cards[card]))

            if rank_list[0] <= rank_list[1] and rank_list[0] <= rank_list[2]:
                return DiscardAction(0)
            elif rank_list[1] <= rank_list[2]:
                return DiscardAction(1)
            else:
                return DiscardAction(2)
            
        strong_cards = "TJQKA"

        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()
            min_cost = min_raise - my_pip
            max_cost = max_raise - my_pip

            is_strong = True
            for card in my_cards:
                if not (_card_rank_char(card) in strong_cards):
                    is_strong = False
                    break

            if is_strong:
                return RaiseAction(min(min_raise * 10, max_raise))
            
            else:
                if random.random() < 0.5:
                    return RaiseAction(min_raise)
                
        if CheckAction in legal_actions:
            return CheckAction()
        if random.random() < 0.25:
            return FoldAction()
        return CallAction()

class TossHighestCardBot:
    
    def __init__(self):
        self.kept_cards = None
    
    def handle_new_round(self, game_state, round_state, active):
        self.kept_cards = None
    
    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        my_cards = round_state.hands[active]
        
        if DiscardAction in legal_actions:
            ranks = "23456789TJQKA"
            rank_list = [-1, -1, -1]
            
            for i in range(3):
                rank_list[i] = ranks.index(str(my_cards[i])[0])
            
            if rank_list[0] >= rank_list[1] and rank_list[0] >= rank_list[2]:
                return DiscardAction(0)
            elif rank_list[1] >= rank_list[2]:
                return DiscardAction(1)
            else:
                return DiscardAction(2)
        
        if self.kept_cards is None and len(my_cards) == 2:
            self.kept_cards = my_cards
        
        strong_cards = "TJQKA"
        
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()
            
            is_strong = True
            for card in my_cards:
                if not (str(card)[0] in strong_cards):
                    is_strong = False
                    break

            if is_strong:
                return RaiseAction(min(min_raise * 10, max_raise))
            else:
                if random.random() < 0.3:
                    return RaiseAction(min_raise)
        
        if CheckAction in legal_actions:
            return CheckAction()
        
        if random.random() < 0.3:
            return FoldAction()
        
        return CallAction()

class RandomDiscardBot:
    
    def __init__(self):
        self.kept_cards = None
    
    def handle_new_round(self, game_state, round_state, active):
        self.kept_cards = None
    
    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        my_cards = round_state.hands[active]
        
        if DiscardAction in legal_actions:
            return DiscardAction(random.randint(0, 2))
        
        if self.kept_cards is None and len(my_cards) == 2:
            self.kept_cards = my_cards
        
        if RaiseAction in legal_actions and random.random() < 0.3:
            min_raise, max_raise = round_state.raise_bounds()
            amount = random.randint(min_raise, max_raise)
            return RaiseAction(amount)
        
        if CheckAction in legal_actions:
            return CheckAction()
        
        if random.random() < 0.5:
            return FoldAction()
        
        return CallAction()

class KeepSuitedBot:
    
    def __init__(self):
        self.kept_cards = None
    
    def handle_new_round(self, game_state, round_state, active):
        self.kept_cards = None
    
    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        my_cards = round_state.hands[active]
        
        if DiscardAction in legal_actions:
            suits = {}
            for i, card in enumerate(my_cards):
                suit = str(card)[1]
                if suit not in suits:
                    suits[suit] = []
                suits[suit].append(i)
            
            max_suit = None
            max_count = 0
            for suit, indices in suits.items():
                if len(indices) > max_count:
                    max_count = len(indices)
                    max_suit = suit
            
            if max_count >= 2:
                for i in range(3):
                    if str(my_cards[i])[1] != max_suit:
                        return DiscardAction(i)
            
            ranks = "23456789TJQKA"
            rank_list = [-1, -1, -1]
            for i in range(3):
                rank_list[i] = ranks.index(str(my_cards[i])[0])
            
            min_index = rank_list.index(min(rank_list))
            return DiscardAction(min_index)
        
        if self.kept_cards is None and len(my_cards) == 2:
            self.kept_cards = my_cards
        
        if self.kept_cards and len(self.kept_cards) == 2:
            suited = str(self.kept_cards[0])[1] == str(self.kept_cards[1])[1]
            if suited and RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                return RaiseAction(min_raise)
        
        if CheckAction in legal_actions:
            return CheckAction()
        
        if random.random() < 0.4:
            return FoldAction()
        
        return CallAction()

class KeepConnectedBot:
    
    def __init__(self):
        self.kept_cards = None
    
    def handle_new_round(self, game_state, round_state, active):
        self.kept_cards = None
    
    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        my_cards = round_state.hands[active]
        
        if DiscardAction in legal_actions:
            ranks = "23456789TJQKA"
            rank_values = []
            
            for card in my_cards:
                rank_values.append(ranks.index(str(card)[0]))
            
            best_combo = None
            best_gap = 100
            
            combos = [(0, 1), (0, 2), (1, 2)]
            for i, j in combos:
                gap = abs(rank_values[i] - rank_values[j])
                if gap < best_gap:
                    best_gap = gap
                    best_combo = (i, j)
            
            discard_idx = 3 - sum(best_combo)
            return DiscardAction(discard_idx)
        
        if self.kept_cards is None and len(my_cards) == 2:
            self.kept_cards = my_cards
        
        if self.kept_cards and len(self.kept_cards) == 2:
            ranks = "23456789TJQKA"
            rank1 = ranks.index(str(self.kept_cards[0])[0])
            rank2 = ranks.index(str(self.kept_cards[1])[0])
            gap = abs(rank1 - rank2)
            
            if gap <= 2 and RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                return RaiseAction(min_raise)
        
        if CheckAction in legal_actions:
            return CheckAction()
        
        if random.random() < 0.4:
            return FoldAction()
        
        return CallAction()

class DiscardMiddleBot:
    
    def __init__(self):
        self.kept_cards = None
    
    def handle_new_round(self, game_state, round_state, active):
        self.kept_cards = None
    
    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        my_cards = round_state.hands[active]
        
        if DiscardAction in legal_actions:
            ranks = "23456789TJQKA"
            rank_values = []
            
            for card in my_cards:
                rank_values.append(ranks.index(str(card)[0]))
            
            sorted_indices = sorted(range(3), key=lambda i: rank_values[i])
            middle_index = sorted_indices[1]
            return DiscardAction(middle_index)
        
        if self.kept_cards is None and len(my_cards) == 2:
            self.kept_cards = my_cards
        
        if RaiseAction in legal_actions and random.random() < 0.4:
            min_raise, max_raise = round_state.raise_bounds()
            return RaiseAction(min_raise)
        
        if CheckAction in legal_actions:
            return CheckAction()
        
        if random.random() < 0.3:
            return FoldAction()
        
        return CallAction()

OPPONENT_POOL = [
    UltraTightBot,
    UltraLooseBot,
    CallingStationBot,
    ManiacBot,
    RandomBot,
    BluffHeavyBot,
    RockBot,
    BalancedBot,
    GoodBot,
    UltraOptimizedBot,
    TossGoodCards,
    TossHighestCardBot,
    RandomDiscardBot,
    KeepSuitedBot,
    KeepConnectedBot,
    DiscardMiddleBot,
    ReferenceBot
]

def _parse_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, '').strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)

class SelfPlaySnapshot:

    def __init__(self):
        here = Path(__file__).resolve().parent
        learner_dir = os.environ.get('LEARNER_BOT_DIR', '').strip() or 'neural_cfr_bot'
        main_bot_player = (here.parent / learner_dir / 'player.py').resolve()

        spec = importlib.util.spec_from_file_location('main_bot_player', str(main_bot_player))
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Could not load spec for {main_bot_player}')

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        BotClass = getattr(module, 'CompleteNeuralCFRBot', None)
        if BotClass is None:
            raise RuntimeError(f'CompleteNeuralCFRBot not found in {main_bot_player}')

        _prev = os.environ.get('NCFR_TRAINING')
        os.environ['NCFR_TRAINING'] = '0'
        try:
            self.bot = BotClass()
        finally:
            if _prev is None:
                os.environ.pop('NCFR_TRAINING', None)
            else:
                os.environ['NCFR_TRAINING'] = _prev
        self.bot.training_mode = False
        try:
            fn = getattr(self.bot, '_sync_model_modes', None)
            if callable(fn):
                fn()
        except Exception:
            pass

    def handle_new_round(self, game_state, round_state, active):
        try:
            return self.bot.handle_new_round(game_state, round_state, active)
        except Exception:
            return None

    def handle_round_over(self, game_state, terminal_state, active):
        try:
            return self.bot.handle_round_over(game_state, terminal_state, active)
        except Exception:
            return None

    def get_action(self, game_state, round_state, active):
        return self.bot.get_action(game_state, round_state, active)

class PoolOpponent(Bot):

    def __init__(self):
        seed = os.environ.get('POOL_SEED', '').strip()
        if seed:
            try:
                random.seed(int(seed))
            except Exception:
                random.seed(seed)
            print(f"[pool] POOL_SEED={seed}", file=sys.stderr, flush=True)

        force = os.environ.get('POOL_OPPONENT', '').strip()
        chosen_cls = None
        if force:
            if force.lower() not in {'selfplay', 'selfplaysnapshot', 'snapshot', 'self'}:
                for cls in OPPONENT_POOL:
                    if cls.__name__.lower() == force.lower():
                        chosen_cls = cls
                        break
                if chosen_cls is not None:
                    print(f"[pool] POOL_OPPONENT={force} -> using {chosen_cls.__name__}", file=sys.stderr, flush=True)
                else:
                    print(f"[pool] POOL_OPPONENT={force} did not match pool; selecting randomly", file=sys.stderr, flush=True)
            else:
                print(f"[pool] POOL_OPPONENT={force} requested, but self-play is disabled here; selecting randomly", file=sys.stderr, flush=True)

        if chosen_cls is None:
            chosen_cls = random.choice(OPPONENT_POOL)
            print(f"[pool] Selected opponent: {chosen_cls.__name__}", file=sys.stderr, flush=True)

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

if __name__ == '__main__':
    run_bot(PoolOpponent(), parse_args())
