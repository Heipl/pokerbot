"""Train a policy-gradient (PPO, or REINFORCE with --algo) policy.

Writes a JSON weights file for the optional PG loader in mccfr_bot/player.py,
which reads it only when GOODBOT_USE_PG=1.

    python tools/train_goodbot_pg.py --episodes 20000
    GOODBOT_USE_PG=1 python engine.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import List, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    if s <= 0:
        n = len(logits)
        return [1.0 / float(n) for _ in range(n)]
    return [e / s for e in exps]


def _softmax_np(logits):
    # logits: shape (K,)
    logits = logits - np.max(logits)
    exps = np.exp(logits)
    s = np.sum(exps)
    if s <= 0:
        return np.full_like(exps, 1.0 / float(exps.shape[0]), dtype=np.float64)
    return exps / s


def _dot(w: List[float], x: List[float]) -> float:
    s = 0.0
    for wi, xi in zip(w, x):
        s += float(wi) * float(xi)
    return s


class MainActionId:
    FOLD = 0
    CALL = 1
    CHECK = 2
    RAISE = 3
    N_ACTIONS = 4


class StepKind:
    MAIN = 0
    DISCARD = 1


# Trajectory step: (kind, feats, candidates, probs, chosen, raise_frac,
# reward_bb, done). reward_bb is booked once per hand, at the hand's end.


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # One "episode" = one full match (multiple rounds/hands).
    p.add_argument('--episodes', type=int, default=50000)
    p.add_argument('--seed', type=int, default=0)
    # More aggressive defaults (faster learning, higher variance). If you see instability,
    # dial these back toward lr=0.001, lr-v=0.002, ppo-epochs=4, ent-coef=0.01.
    p.add_argument('--lr', type=float, default=0.00025)
    p.add_argument('--lr-v', type=float, default=0.0005)
    p.add_argument('--gamma', type=float, default=0.97)
    p.add_argument('--algo', type=str, default='ppo', choices=['reinforce', 'ppo'])
    p.add_argument('--gae-lambda', type=float, default=0.92)
    p.add_argument('--clip-eps', type=float, default=0.15)
    p.add_argument('--ent-coef', type=float, default=0.012)
    p.add_argument('--vf-coef', type=float, default=0.75)
    p.add_argument('--max-grad-norm', type=float, default=0.8)
    p.add_argument('--ppo-epochs', type=int, default=3)
    p.add_argument('--minibatch', type=int, default=512)
    p.add_argument('--rounds', type=int, default=100, help='Hands per match; 0 uses engine.NUM_ROUNDS')
    p.add_argument('--feature-dim', type=int, default=0, help='0 uses pg_features.feature_size()')
    # 128 because every league snapshot was trained at 128 and
    # _load_mlp_params_from_json raises on a dims mismatch.
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--clip-adv', type=float, default=4.0, help='Clip advantage to [-clip_adv, clip_adv]')
    p.add_argument('--clip-w', type=float, default=5.0, help='Clip weights/biases to [-clip_w, clip_w] (0 disables)')
    p.add_argument('--out', type=str, default='mccfr_bot/goodbot_pg_policy.json')
    p.add_argument('--log-every', type=int, default=1)
    p.add_argument('--eval-every', type=int, default=150, help='Evaluate periodically vs heuristic GoodBot (0 disables)')
    p.add_argument('--eval-matches', type=int, default=40, help='Matches per evaluation')
    p.add_argument('--dense-rewards', action=argparse.BooleanOptionalAction, default=True, help='Use dense reward shaping with immediate rewards per action (use --no-dense-rewards to disable)')    

    # --- Checkpointing ---
    p.add_argument(
        '--save-best',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Save best policy checkpoint during training (use --no-save-best to disable)',
    )
    p.add_argument('--best-metric', type=str, default='eval', choices=['eval', 'ema'], help='Metric used when --save-best is set')
    p.add_argument('--best-out', type=str, default='', help='Path for best checkpoint JSON (default: derived from --out)')

    # --- Resume / initialization ---
    p.add_argument(
        '--init-from-best',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Initialize learner weights from the current best checkpoint (derived from --out/--best-metric) if it exists (use --no-init-from-best to disable)',
    )
    p.add_argument(
        '--init-from',
        type=str,
        default='',
        help='Explicit path to a checkpoint JSON to initialize learner weights from (overrides --init-from-best)',
    )

    # --- Opponent mixing ---
    p.add_argument(
        '--train-vs-heuristic-prob',
        type=float,
        default=0.2,
        help='With this probability, train episode vs heuristic GoodBot instead of a snapshot (0 disables)',
    )

    p.add_argument(
        '--train-vs-pool-prob',
        type=float,
        default=0.8,
        help='With this probability, train episode vs opponent_pool PoolOpponent instead of a snapshot (0 disables)',
    )

    # opponent_pool selection when --train-vs-pool-prob > 0. The pool is
    # reseeded per episode from --seed: varied opponents, still reproducible.
    p.add_argument('--pool-seed', type=int, default=-1, help='If >=0, sets POOL_SEED to this fixed value for PoolOpponent; -1 reseeds per episode')
    p.add_argument('--pool-opponent', type=str, default='', help='If set, forces POOL_OPPONENT for PoolOpponent (e.g., GoodBot, ManiacBot)')
    p.add_argument(
        '--pool-sampling',
        type=str,
        default='shuffle',
        choices=['random', 'shuffle', 'roundrobin'],
        help='When training vs opponent_pool: random=let PoolOpponent pick; shuffle/roundrobin=force POOL_OPPONENT to cover the whole pool over time (ignored if --pool-opponent is set)',
    )

    # Which bot folder to train features for (must contain pg_features.py).
    # Examples: opponent_pool, mccfr_bot
    p.add_argument('--bot-dir', type=str, default='mccfr_bot', help='Bot folder (relative to engine root) that contains pg_features.py')

    # --- League / opponent snapshot training ---
    # Maintains a pool of frozen opponent snapshots and samples from it each episode.
    p.add_argument('--league-dir', type=str, default='opponent_pool/pg_league_pool')
    p.add_argument('--freeze-every', type=int, default=15, help='Freeze current policy into league pool every N episodes (0 disables)')
    p.add_argument('--max-league-size', type=int, default=30, help='Max number of snapshots to keep (0 keeps all)')
    p.add_argument('--opp-deterministic', action='store_true', help='Opponent plays deterministically (argmax/mean sizing)')
    p.add_argument(
        '--randomize-seat',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Randomize whether learner is seat 0 or 1 each episode (use --no-randomize-seat to disable)',
    )
    return p

def _calculate_immediate_reward(old_rs, action, new_rs, active: int, street: int) -> float:
    reward = 0.0

    try:
        import engine
        STARTING_STACK = engine.STARTING_STACK
        BIG_BLIND = engine.BIG_BLIND
        my_chips_before = int(old_rs.stacks[active])
        my_chips_after = int(new_rs.stacks[active])
        chip_delta = float(my_chips_after - my_chips_before)
        
        if abs(chip_delta) > 1e-6:
            reward += chip_delta / float(BIG_BLIND) * 0.5
        
        # Pot control.
        pot_before = sum(int(STARTING_STACK - s) for s in old_rs.stacks)
        my_pip_before = int(old_rs.pips[active])
        my_pip_after = int(new_rs.pips[active])
        pip_increase = my_pip_after - my_pip_before
        
        if pip_increase > 0:
            pot_ratio = float(pip_increase) / max(1.0, float(pot_before))
            if pot_ratio > 0.75 and street < 5:  # Overbetting early streets
                reward -= 0.1 * pot_ratio
            elif pot_ratio < 0.25 and street >= 4:  # Underbetting value on later streets
                reward -= 0.05
        
        # Position bonus.
        if street == 0 and active == 1:  # Preflop, button
            reward += 0.03
        elif street >= 4 and active == 0:  # Postflop, in position
            reward += 0.02
        
        # Aggression bonus.
        if isinstance(action, engine.RaiseAction):
            min_raise, max_raise = old_rs.raise_bounds()
            raise_amount = pip_increase
            if max_raise > min_raise:
                raise_fraction = float(raise_amount - min_raise) / float(max_raise - min_raise)
                if 0.3 <= raise_fraction <= 0.7:
                    reward += 0.04
                elif raise_fraction > 0.9:
                    reward -= 0.03
        
        # Won without showdown: fold equity worked.
        if isinstance(new_rs, engine.TerminalState) and chip_delta > 0:
            reward += 0.1
        
        reward = max(-0.5, min(0.5, reward))
        
    except Exception:
        pass
    
    return float(reward)


def _calculate_hand_outcome_reward(rs, active: int, hand_actions: List[Tuple], hand_idx: int) -> float:
    try:
        import engine
        hand_delta = float(rs.deltas[active]) / float(engine.BIG_BLIND)
        
        total_actions = len(hand_actions)
        if total_actions == 0:
            return 0.0
        
        # Later actions get more credit for the outcome.
        position_weight = float(hand_idx + 1) / float(total_actions)
        return hand_delta * position_weight * 0.3  # Scale down to avoid overpowering immediate rewards
        
    except Exception:
        return 0.0

def main() -> int:
    args = build_argparser().parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if root not in sys.path:
        sys.path.insert(0, root)

    import pkrbot  # noqa: F401
    import engine

    # Shared feature extractor (must match runtime for the bot you're training).
    bot_dir = os.path.join(root, str(args.bot_dir)) if not os.path.isabs(str(args.bot_dir)) else str(args.bot_dir)
    bot_dir = os.path.abspath(bot_dir)
    pg_features_path = os.path.join(bot_dir, 'pg_features.py')
    if not os.path.isfile(pg_features_path):
        raise RuntimeError(f'pg_features.py not found at: {pg_features_path} (set --bot-dir)')

    # The bot dir must be importable, not just readable: pg_features does
    # `from mccfr_common import ...`, and loading it by file path alone leaves
    # that import to fail into its zero-filled fallback -- training on four
    # features the running bot computes for real.
    if bot_dir not in sys.path:
        sys.path.insert(0, bot_dir)

    import importlib.util

    spec = importlib.util.spec_from_file_location('pg_features', pg_features_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load pg_features spec from: {pg_features_path}')
    pg_features = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pg_features)  # type: ignore[attr-defined]

    pg_extract_features = getattr(pg_features, 'extract_features')
    pg_feature_size = getattr(pg_features, 'feature_size')

    rng = random.Random(args.seed)

    STARTING_STACK = engine.STARTING_STACK
    BIG_BLIND = engine.BIG_BLIND
    NUM_ROUNDS = int(engine.NUM_ROUNDS)

    feature_dim = int(args.feature_dim) if int(args.feature_dim) > 0 else int(pg_feature_size())

    def _atomic_write_json(path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = str(path) + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(payload, f)
        try:
            os.replace(tmp, path)
        except Exception:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f)
            try:
                os.remove(tmp)
            except Exception:
                pass

    def extract_features(round_state, active: int, dim: int = 0) -> List[float]:
        # `dim` feeds an opponent policy the width it was trained at: new
        # features are appended at the end, so truncating to 184 gives a
        # 184-dim snapshot exactly the vector it learned on.
        return pg_extract_features(
            round_state,
            active,
            starting_stack=int(STARTING_STACK),
            big_blind=int(BIG_BLIND),
            feature_dim=int(dim) if int(dim) > 0 else int(feature_dim),
        )

    def candidate_main_action_ids(round_state, active: int) -> List[int]:
        legal = round_state.legal_actions()
        ids: List[int] = []
        if engine.FoldAction in legal:
            ids.append(MainActionId.FOLD)
        if engine.CallAction in legal:
            ids.append(MainActionId.CALL)
        if engine.CheckAction in legal:
            ids.append(MainActionId.CHECK)
        if engine.RaiseAction in legal:
            ids.append(MainActionId.RAISE)
        return ids

    def candidate_discard_action_ids(round_state, active: int) -> List[int]:
        my_cards = list(round_state.hands[active] or [])
        return list(range(min(3, len(my_cards))))

    def _raise_amount_from_fraction(round_state, active: int, f: float) -> int:
        min_raise, max_raise = round_state.raise_bounds()
        if int(max_raise) <= int(min_raise):
            return int(min_raise)
        f = float(max(0.0, min(1.0, float(f))))
        amt = int(round(float(min_raise) + f * float(int(max_raise) - int(min_raise))))
        if amt < int(min_raise):
            amt = int(min_raise)
        if amt > int(max_raise):
            amt = int(max_raise)

        # Match runtime safety cap for preflop.
        try:
            street = int(getattr(round_state, 'street', 0) or 0)
        except Exception:
            street = 0
        try:
            my_pip = int(round_state.pips[active])
        except Exception:
            my_pip = 0
        try:
            cap_bb = float(os.environ.get('GOODBOT_PG_PREFLOP_RAISE_CAP_BB', '12') or '12')
        except Exception:
            cap_bb = 12.0
        if street == 0 and cap_bb > 0:
            cap_amount = int(my_pip + cap_bb * float(BIG_BLIND))
            if cap_amount >= int(min_raise):
                amt = int(min(amt, cap_amount))

        return int(amt)

    use_numpy = (np is not None)
    if args.algo == 'ppo' and not use_numpy:
        raise RuntimeError('PPO trainer requires numpy; please install numpy or use --algo reinforce')
    if use_numpy:
        hidden = int(args.hidden)
        np_rng = np.random.default_rng(int(args.seed) if int(args.seed) != 0 else None)
        init_scale = 0.01
        W1 = (init_scale * np_rng.standard_normal((hidden, feature_dim))).astype(np.float64)
        b1 = np.zeros((hidden,), dtype=np.float64)
        Wp_main = (init_scale * np_rng.standard_normal((MainActionId.N_ACTIONS, hidden))).astype(np.float64)
        bp_main = np.zeros((MainActionId.N_ACTIONS,), dtype=np.float64)
        Wp_discard = (init_scale * np_rng.standard_normal((3, hidden))).astype(np.float64)
        bp_discard = np.zeros((3,), dtype=np.float64)
        w_beta_a = (init_scale * np_rng.standard_normal((hidden,))).astype(np.float64)
        b_beta_a = 0.0
        w_beta_b = (init_scale * np_rng.standard_normal((hidden,))).astype(np.float64)
        b_beta_b = 0.0
        Wv = (init_scale * np_rng.standard_normal((hidden,))).astype(np.float64)
        bv = 0.0

        def _relu(x):
            return np.maximum(0.0, x)

        def _softplus(x: float) -> float:
            # scalar softplus
            if x > 50.0:
                return float(x)
            if x < -50.0:
                return float(np.exp(x))
            return float(np.log1p(np.exp(x)))

        def _sigmoid(x: float) -> float:
            if x >= 0.0:
                z = math.exp(-x)
                return 1.0 / (1.0 + z)
            z = math.exp(x)
            return z / (1.0 + z)

        def forward(x):
            hpre = (W1 @ x) + b1
            h = _relu(hpre)
            logits_main = (Wp_main @ h) + bp_main
            logits_discard = (Wp_discard @ h) + bp_discard
            raw_a = float(w_beta_a @ h + b_beta_a)
            raw_b = float(w_beta_b @ h + b_beta_b)
            v = float(Wv @ h + bv)
            return hpre, h, logits_main, logits_discard, raw_a, raw_b, v

        def value(feats) -> float:
            x = np.asarray(feats, dtype=np.float64)
            _hpre, _h, _logits_main, _logits_discard, _raw_a, _raw_b, v = forward(x)
            return float(v)
            
        
    else:
        # Initialize weights (pure python fallback)
        W_main = [[0.0 for _ in range(feature_dim)] for _ in range(MainActionId.N_ACTIONS)]
        b_main = [0.0 for _ in range(MainActionId.N_ACTIONS)]
        W_discard = [[0.0 for _ in range(feature_dim)] for _ in range(3)]
        b_discard = [0.0 for _ in range(3)]
        w_beta_a = [0.0 for _ in range(feature_dim)]
        b_beta_a = 0.0
        w_beta_b = [0.0 for _ in range(feature_dim)]
        b_beta_b = 0.0
        v_w = [0.0 for _ in range(feature_dim)]
        v_b = 0.0

        def value(feats: List[float]) -> float:
            return _dot(v_w, feats) + v_b

    # --- League training helpers (numpy/MLP path) ---
    class _MLPParams:
        def __init__(
            self,
            feature_dim: int,
            hidden: int,
            W1,
            b1,
            Wp_main,
            bp_main,
            Wp_discard,
            bp_discard,
            w_beta_a,
            b_beta_a,
            w_beta_b,
            b_beta_b,
            Wv,
            bv,
        ):
            self.feature_dim = int(feature_dim)
            self.hidden = int(hidden)
            self.W1 = W1
            self.b1 = b1
            self.Wp_main = Wp_main
            self.bp_main = bp_main
            self.Wp_discard = Wp_discard
            self.bp_discard = bp_discard
            self.w_beta_a = w_beta_a
            self.b_beta_a = float(b_beta_a)
            self.w_beta_b = w_beta_b
            self.b_beta_b = float(b_beta_b)
            self.Wv = Wv
            self.bv = float(bv)

    def _forward_params(params: _MLPParams, x):
        hpre = (params.W1 @ x) + params.b1
        h = np.maximum(0.0, hpre)
        logits_main = (params.Wp_main @ h) + params.bp_main
        logits_discard = (params.Wp_discard @ h) + params.bp_discard
        raw_a = float(params.w_beta_a @ h + params.b_beta_a)
        raw_b = float(params.w_beta_b @ h + params.b_beta_b)
        v = float(params.Wv @ h + params.bv)
        return hpre, h, logits_main, logits_discard, raw_a, raw_b, v

    def _payload_from_current(meta: dict = None) -> dict:
        payload = {
            'feature_dim': feature_dim,
            'model': 'mlp' if use_numpy else 'linear',
            'action_space': 'param_raise_v1',
            'hidden': int(args.hidden) if use_numpy else None,
            'W1': W1.tolist() if use_numpy else None,
            'b1': b1.tolist() if use_numpy else None,
            'Wp_main': Wp_main.tolist() if use_numpy else None,
            'bp_main': bp_main.tolist() if use_numpy else None,
            'Wp_discard': Wp_discard.tolist() if use_numpy else None,
            'bp_discard': bp_discard.tolist() if use_numpy else None,
            'w_beta_a': w_beta_a.tolist() if use_numpy else w_beta_a,
            'b_beta_a': b_beta_a if use_numpy else b_beta_a,
            'w_beta_b': w_beta_b.tolist() if use_numpy else w_beta_b,
            'b_beta_b': b_beta_b if use_numpy else b_beta_b,
            'Wv': Wv.tolist() if use_numpy else None,
            'bv': bv if use_numpy else None,
            'W_main': W_main if (not use_numpy) else None,
            'b_main': b_main if (not use_numpy) else None,
            'W_discard': W_discard if (not use_numpy) else None,
            'b_discard': b_discard if (not use_numpy) else None,
            'v_w': v_w.tolist() if (not use_numpy) else None,
            'v_b': v_b if (not use_numpy) else None,
            'algo': str(args.algo),
            'main_action_names': ['FOLD', 'CALL', 'CHECK', 'RAISE'],
            'discard_action_names': ['DISCARD_0', 'DISCARD_1', 'DISCARD_2'],
        }
        if isinstance(meta, dict) and meta:
            payload['meta'] = meta
        return payload

    def _load_mlp_params_from_json(path: str, exact_dims: bool = True) -> _MLPParams:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if str(data.get('action_space', '')).strip() != 'param_raise_v1':
            raise RuntimeError(f'Unsupported action_space in snapshot: {data.get("action_space")}')
        if str(data.get('model', '')).strip() != 'mlp':
            raise RuntimeError(f'Unsupported model in snapshot (expected mlp): {data.get("model")}')

        fdim = int(data.get('feature_dim', 0) or 0)
        hid = int(data.get('hidden', 0) or 0)
        # exact_dims=False for opponent snapshots: they run on their own array
        # shapes, so only a snapshot wanting MORE features than pg_features has
        # is unusable. It stays True for _try_init_learner_from, which copies
        # into the learner's fixed-shape arrays.
        bad = ((fdim != int(feature_dim) or hid != int(args.hidden)) if exact_dims
               else (fdim <= 0 or fdim > int(feature_dim)))
        if bad:
            # Raise rather than skip: a silently ignored opponent is worse.
            # The usual cause is a changed pg_features.feature_size(), which
            # makes every snapshot stale and unconvertible.
            raise RuntimeError(
                f'Snapshot dims mismatch in {os.path.basename(path)}: '
                f'snapshot feature_dim={fdim} hidden={hid}, '
                f'this run needs feature_dim={int(feature_dim)} hidden={int(args.hidden)}.\n'
                f'  If feature_dim differs, pg_features.py changed and the league is stale:\n'
                f'    rm -rf opponent_pool/pg_league_pool/*.json\n'
                f'  If only hidden differs, pass --hidden {hid} to match the league.')

        # Validate against the SNAPSHOT's own dims when it is being loaded as an
        # opponent; only the exact-dims path (learner init) must match this run.
        exp_dim = int(feature_dim) if exact_dims else fdim
        exp_hid = int(args.hidden) if exact_dims else hid

        def arr(a, shape=None):
            out = np.asarray(a, dtype=np.float64)
            if shape is not None and tuple(out.shape) != tuple(shape):
                raise RuntimeError(f'Snapshot array shape mismatch, expected {shape}, got {out.shape}')
            return out

        W1s = arr(data['W1'], (exp_hid, exp_dim))
        b1s = arr(data['b1'], (exp_hid,))
        Wp_m = arr(data['Wp_main'], (MainActionId.N_ACTIONS, exp_hid))
        bp_m = arr(data['bp_main'], (MainActionId.N_ACTIONS,))
        Wp_d = arr(data['Wp_discard'], (3, exp_hid))
        bp_d = arr(data['bp_discard'], (3,))
        w_a = arr(data['w_beta_a'], (exp_hid,))
        b_a = float(data['b_beta_a'])
        w_b = arr(data['w_beta_b'], (exp_hid,))
        b_b = float(data['b_beta_b'])
        Wv_s = arr(data.get('Wv', np.zeros((exp_hid,), dtype=np.float64)), (exp_hid,))
        bv_s = float(data.get('bv', 0.0))
        return _MLPParams(
            # The policy's OWN width/height, not this run's. _choose_action_with_params
            # reads these back to feed the network the vector it was trained on.
            feature_dim=exp_dim,
            hidden=exp_hid,
            W1=W1s,
            b1=b1s,
            Wp_main=Wp_m,
            bp_main=bp_m,
            Wp_discard=Wp_d,
            bp_discard=bp_d,
            w_beta_a=w_a,
            b_beta_a=b_a,
            w_beta_b=w_b,
            b_beta_b=b_b,
            Wv=Wv_s,
            bv=bv_s,
        )

    def _try_init_learner_from(path: str) -> bool:
        """Load weights from a checkpoint JSON into the current learner arrays."""
        nonlocal b_beta_a, b_beta_b, bv
        if not use_numpy:
            return False
        try:
            if not path:
                return False
            if not os.path.isabs(path):
                path = os.path.join(root, path)
            if not os.path.isfile(path):
                return False
            src = _load_mlp_params_from_json(path)
            W1[:] = src.W1
            b1[:] = src.b1
            Wp_main[:] = src.Wp_main
            bp_main[:] = src.bp_main
            Wp_discard[:] = src.Wp_discard
            bp_discard[:] = src.bp_discard
            w_beta_a[:] = src.w_beta_a
            b_beta_a = float(src.b_beta_a)
            w_beta_b[:] = src.w_beta_b
            b_beta_b = float(src.b_beta_b)
            Wv[:] = src.Wv
            bv = float(src.bv)
            print(f'init: loaded learner weights from {os.path.abspath(path)}')
            return True
        except Exception as e:
            try:
                print(f'init: failed to load weights from {path}: {e}')
            except Exception:
                pass
            return False

    def _list_league_snapshots(league_dir: str) -> List[str]:
        if not os.path.isdir(league_dir):
            return []
        paths = []
        for name in os.listdir(league_dir):
            if not name.lower().endswith('.json'):
                continue
            if name.lower().startswith('manifest'):
                continue
            paths.append(os.path.join(league_dir, name))
        paths.sort()
        return paths

    def _prune_league(league_dir: str, max_size: int) -> None:
        if int(max_size) <= 0:
            return
        snaps = _list_league_snapshots(league_dir)
        if len(snaps) <= int(max_size):
            return
        # Remove oldest by name ordering.
        to_remove = snaps[: max(0, len(snaps) - int(max_size))]
        for pth in to_remove:
            try:
                os.remove(pth)
            except Exception:
                pass

    def _freeze_to_league(league_dir: str, episode: int) -> str:
        os.makedirs(league_dir, exist_ok=True)
        fname = f'snapshot_ep{int(episode):07d}.json'
        path = os.path.join(league_dir, fname)
        tmp = path + '.tmp'
        payload = _payload_from_current()
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(payload, f)
        try:
            os.replace(tmp, path)
        except Exception:
            # Best-effort fallback.
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f)
            try:
                os.remove(tmp)
            except Exception:
                pass
        return path

    def _choose_action_with_params(rs, active: int, params: _MLPParams, deterministic: bool, record_traj: bool, traj_out):
        feats = extract_features(rs, active, dim=int(getattr(params, 'feature_dim', 0) or 0))
        legal = rs.legal_actions()

        if engine.DiscardAction in legal:
            candidates = candidate_discard_action_ids(rs, active)
            if not candidates:
                a = engine.DiscardAction(0)
                return a
            x = np.asarray(feats, dtype=np.float64)
            _hpre, _h, _logits_main, logits_discard, _raw_a, _raw_b, _v = _forward_params(params, x)
            cand = np.asarray(candidates, dtype=np.int64)
            logits = logits_discard[cand]
            probs = _softmax_np(logits)
            probs_list = probs.tolist()
            if deterministic:
                idx = int(np.argmax(probs))
            else:
                idx = int(np_rng.choice(len(candidates), p=probs))
            choice = int(candidates[idx])
            if record_traj and traj_out is not None:
                traj_out.append((StepKind.DISCARD, feats, candidates, probs_list, int(choice), float('nan'), 0.0, 0.0))
            return engine.DiscardAction(int(choice))

        candidates = candidate_main_action_ids(rs, active)
        if not candidates:
            if engine.CheckAction in legal:
                return engine.CheckAction()
            if engine.CallAction in legal:
                return engine.CallAction()
            return engine.FoldAction()

        x = np.asarray(feats, dtype=np.float64)
        _hpre, _h, logits_main, _logits_discard, raw_a, raw_b, _v = _forward_params(params, x)
        cand = np.asarray(candidates, dtype=np.int64)
        logits = logits_main[cand]
        probs = _softmax_np(logits)
        probs_list = probs.tolist()

        if deterministic:
            idx = int(np.argmax(probs))
        else:
            idx = int(np_rng.choice(len(candidates), p=probs))
        chosen = int(candidates[idx])

        f = float('nan')
        if chosen == MainActionId.RAISE and engine.RaiseAction in legal:
            alpha = _softplus(float(raw_a)) + 1e-3
            beta = _softplus(float(raw_b)) + 1e-3
            f = float(alpha / max(1e-12, (alpha + beta))) if deterministic else float(np_rng.beta(alpha, beta))
            amt = _raise_amount_from_fraction(rs, active, f)
            act = engine.RaiseAction(int(amt))
        elif chosen == MainActionId.FOLD and engine.FoldAction in legal:
            act = engine.FoldAction()
        elif chosen == MainActionId.CALL and engine.CallAction in legal:
            act = engine.CallAction()
        elif chosen == MainActionId.CHECK and engine.CheckAction in legal:
            act = engine.CheckAction()
        else:
            if engine.CheckAction in legal:
                act = engine.CheckAction()
            elif engine.CallAction in legal:
                act = engine.CallAction()
            else:
                act = engine.FoldAction()

        if record_traj and traj_out is not None:
            traj_out.append((StepKind.MAIN, feats, candidates, probs_list, int(chosen), float(f), 0.0, 0.0))
        return act

    # PPO needs numpy; without it we fall back to mirror self-play.
    learner_params = None
    if use_numpy:
        learner_params = _MLPParams(
            feature_dim=int(feature_dim),
            hidden=int(args.hidden),
            W1=W1,
            b1=b1,
            Wp_main=Wp_main,
            bp_main=bp_main,
            Wp_discard=Wp_discard,
            bp_discard=bp_discard,
            w_beta_a=w_beta_a,
            b_beta_a=b_beta_a,
            w_beta_b=w_beta_b,
            b_beta_b=b_beta_b,
            Wv=Wv,
            bv=bv,
        )

    def play_match_mirror() -> Tuple[List[float], List[List[Tuple[int, List[float], List[int], List[float], int, float, float]]]]:
        """Self-play match, both seats on the current policy. Fallback for when
        league training is off or numpy is missing."""
        rounds = int(args.rounds) if int(args.rounds) > 0 else NUM_ROUNDS
        bankroll = [0.0, 0.0]
        traj: List[List[Tuple[int, List[float], List[int], List[float], int, float, float]]] = [[], []]

        for _ in range(rounds):
            deck = engine.pkrbot.Deck()
            deck.shuffle()
            hands = [deck.deal(3), deck.deal(3)]
            board = []
            pips = [engine.SMALL_BLIND, engine.BIG_BLIND]
            stacks = [engine.STARTING_STACK - engine.SMALL_BLIND, engine.STARTING_STACK - engine.BIG_BLIND]
            rs = engine.RoundState(0, 0, pips, stacks, hands, deck, board, None)

            while not isinstance(rs, engine.TerminalState):
                active = rs.button % 2
                legal = rs.legal_actions()

                if use_numpy:
                    # Sample stochastically for both seats.
                    act = _choose_action_with_params(rs, active, learner_params, deterministic=False, record_traj=True, traj_out=traj[active])
                    rs = rs.proceed(act)
                    continue

                # Pure python fallback (no beta head update; uniform sizing).
                feats = extract_features(rs, active)
                if engine.DiscardAction in legal:
                    candidates = candidate_discard_action_ids(rs, active)
                    logits = [_dot(W_discard[a], feats) + float(b_discard[a]) for a in candidates]
                    probs_list = _softmax(logits)
                    r = rng.random()
                    csum = 0.0
                    idx = len(probs_list) - 1
                    for i, p in enumerate(probs_list):
                        csum += float(p)
                        if r <= csum:
                            idx = i
                            break
                    chosen = int(candidates[idx]) if candidates else 0
                    traj[active].append((StepKind.DISCARD, feats, candidates, probs_list, chosen, float('nan'), 0.0, 0.0))
                    rs = rs.proceed(engine.DiscardAction(int(chosen)))
                    continue

                candidates = candidate_main_action_ids(rs, active)
                logits = [_dot(W_main[a], feats) + float(b_main[a]) for a in candidates]
                probs_list = _softmax(logits)
                r = rng.random()
                csum = 0.0
                idx = len(probs_list) - 1
                for i, p in enumerate(probs_list):
                    csum += float(p)
                    if r <= csum:
                        idx = i
                        break
                chosen = int(candidates[idx]) if candidates else MainActionId.CHECK

                f = float('nan')
                if chosen == MainActionId.RAISE and engine.RaiseAction in legal:
                    f = rng.random()
                    amt = _raise_amount_from_fraction(rs, active, f)
                    rs = rs.proceed(engine.RaiseAction(int(amt)))
                else:
                    if chosen == MainActionId.FOLD and engine.FoldAction in legal:
                        rs = rs.proceed(engine.FoldAction())
                    elif chosen == MainActionId.CALL and engine.CallAction in legal:
                        rs = rs.proceed(engine.CallAction())
                    elif chosen == MainActionId.CHECK and engine.CheckAction in legal:
                        rs = rs.proceed(engine.CheckAction())
                    else:
                        if engine.CheckAction in legal:
                            rs = rs.proceed(engine.CheckAction())
                        elif engine.CallAction in legal:
                            rs = rs.proceed(engine.CallAction())
                        else:
                            rs = rs.proceed(engine.FoldAction())

                traj[active].append((StepKind.MAIN, feats, candidates, probs_list, chosen, float(f), 0.0, 0.0))

            bankroll[0] += float(rs.deltas[0])
            bankroll[1] += float(rs.deltas[1])

            # Dense reward shaping: assign this hand's delta (in BB) to the
            # last recorded action for each seat, if any.
            try:
                for seat in (0, 1):
                    if traj[int(seat)]:
                        k, feats, cands, probs, chosen, frac, r_old, _d0 = traj[int(seat)][-1]
                        traj[int(seat)][-1] = (k, feats, cands, probs, chosen, frac, float(r_old) + float(rs.deltas[int(seat)]) / float(BIG_BLIND), 1.0)
            except Exception:
                pass

        return [bankroll[0] / float(BIG_BLIND), bankroll[1] / float(BIG_BLIND)], traj

    def play_match_vs_snapshot(opponent_params: _MLPParams, learner_seat: int) -> Tuple[List[float], List[List[Tuple[int, List[float], List[int], List[float], int, float, float]]]]:
        rounds = int(args.rounds) if int(args.rounds) > 0 else NUM_ROUNDS
        bankroll = [0.0, 0.0]
        traj: List[List[Tuple[int, List[float], List[int], List[float], int, float, float]]] = [[], []]
        opp_det = bool(getattr(args, 'opp_deterministic', False))

        for _ in range(rounds):
            deck = engine.pkrbot.Deck()
            deck.shuffle()
            hands = [deck.deal(3), deck.deal(3)]
            board = []
            pips = [engine.SMALL_BLIND, engine.BIG_BLIND]
            stacks = [engine.STARTING_STACK - engine.SMALL_BLIND, engine.STARTING_STACK - engine.BIG_BLIND]
            rs = engine.RoundState(0, 0, pips, stacks, hands, deck, board, None)

            while not isinstance(rs, engine.TerminalState):
                active = rs.button % 2
                if int(active) == int(learner_seat):
                    act = _choose_action_with_params(rs, active, learner_params, deterministic=False, record_traj=True, traj_out=traj[learner_seat])
                else:
                    act = _choose_action_with_params(rs, active, opponent_params, deterministic=opp_det, record_traj=False, traj_out=None)
                rs = rs.proceed(act)

            bankroll[0] += float(rs.deltas[0])
            bankroll[1] += float(rs.deltas[1])

            # Assign learner's hand delta to its last recorded action (if any).
            try:
                ls = int(learner_seat)
                if traj[ls]:
                    k, feats, cands, probs, chosen, frac, r_old, _d0 = traj[ls][-1]
                    traj[ls][-1] = (k, feats, cands, probs, chosen, frac, float(r_old) + float(rs.deltas[ls]) / float(BIG_BLIND), 1.0)
            except Exception:
                pass


        # Reward is final bankroll over the full match in BB units for the learner seat.
        returns = [0.0, 0.0]
        returns[int(learner_seat)] = float(bankroll[int(learner_seat)]) / float(BIG_BLIND)
        return returns, traj

    # --- Evaluation vs heuristic GoodBot (periodic) ---
    class _EvalGS:
        def __init__(self):
            self.bankroll = 0
            # Some opponent_pool bots read game_clock; keep it stable.
            self.game_clock = 999999.0
            self.round_num = 1

    def _convert_heuristic_action_to_engine(a):
        # opponent_pool GoodBot returns skeleton action objects; convert by name.
        name = type(a).__name__
        if name == 'FoldAction':
            return engine.FoldAction()
        if name == 'CallAction':
            return engine.CallAction()
        if name == 'CheckAction':
            return engine.CheckAction()
        if name == 'RaiseAction':
            return engine.RaiseAction(int(getattr(a, 'amount', 0) or 0))
        if name == 'DiscardAction':
            return engine.DiscardAction(int(getattr(a, 'card', 0) or 0))
        # fallback
        legal = getattr(a, 'legal_actions', None)
        return engine.CheckAction()

    def _policy_choose_action(rs, active: int, deterministic: bool = True):
        feats = extract_features(rs, active)
        legal = rs.legal_actions()

        if engine.DiscardAction in legal:
            candidates = candidate_discard_action_ids(rs, active)
            if not candidates:
                return engine.DiscardAction(0)
            x = np.asarray(feats, dtype=np.float64)
            _hpre, _h, _logits_main, logits_discard, _raw_a, _raw_b, _v = forward(x)
            cand = np.asarray(candidates, dtype=np.int64)
            logits = logits_discard[cand]
            probs = _softmax_np(logits)
            idx = int(np.argmax(probs)) if deterministic else int(np_rng.choice(len(candidates), p=probs))
            choice = int(candidates[idx])
            return engine.DiscardAction(choice)

        candidates = candidate_main_action_ids(rs, active)
        if not candidates:
            if engine.CheckAction in legal:
                return engine.CheckAction()
            if engine.CallAction in legal:
                return engine.CallAction()
            return engine.FoldAction()

        x = np.asarray(feats, dtype=np.float64)
        _hpre, _h, logits_main, _logits_discard, raw_a, raw_b, _v = forward(x)
        cand = np.asarray(candidates, dtype=np.int64)
        logits = logits_main[cand]
        probs = _softmax_np(logits)
        idx = int(np.argmax(probs)) if deterministic else int(np_rng.choice(len(candidates), p=probs))
        chosen = int(candidates[idx])

        if chosen == MainActionId.FOLD and engine.FoldAction in legal:
            return engine.FoldAction()
        if chosen == MainActionId.CALL and engine.CallAction in legal:
            return engine.CallAction()
        if chosen == MainActionId.CHECK and engine.CheckAction in legal:
            return engine.CheckAction()
        if chosen == MainActionId.RAISE and engine.RaiseAction in legal:
            alpha = _softplus(float(raw_a)) + 1e-3
            beta = _softplus(float(raw_b)) + 1e-3
            f = float(alpha / max(1e-12, (alpha + beta))) if deterministic else float(np_rng.beta(alpha, beta))
            amt = _raise_amount_from_fraction(rs, active, f)
            return engine.RaiseAction(int(amt))

        if engine.CheckAction in legal:
            return engine.CheckAction()
        if engine.CallAction in legal:
            return engine.CallAction()
        return engine.FoldAction()

    _heuristic_bot = None
    _pool_mod = None
    _pool_names_cache = None
    _pool_shuffle_bag: List[str] = []
    _pool_rr_index = 0
    _last_pool_opp_name = ''

    opp_pool_dir = os.path.join(root, 'opponent_pool')

    def _load_pool_module():
        nonlocal _pool_mod
        if _pool_mod is not None:
            return _pool_mod

        try:
            import importlib.util

            pool_path = os.path.join(opp_pool_dir, 'player.py')
            if not os.path.isfile(pool_path):
                _pool_mod = None
                return None

            spec = importlib.util.spec_from_file_location('pool_player_for_training', pool_path)
            if spec is None or spec.loader is None:
                _pool_mod = None
                return None

            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]

            # Patch action symbols/constants so legality checks work with engine.RoundState.
            try:
                setattr(mod, 'FoldAction', engine.FoldAction)
                setattr(mod, 'CallAction', engine.CallAction)
                setattr(mod, 'CheckAction', engine.CheckAction)
                setattr(mod, 'RaiseAction', engine.RaiseAction)
                setattr(mod, 'DiscardAction', engine.DiscardAction)
                setattr(mod, 'STARTING_STACK', int(engine.STARTING_STACK))
                setattr(mod, 'BIG_BLIND', int(engine.BIG_BLIND))
                setattr(mod, 'SMALL_BLIND', int(engine.SMALL_BLIND))
                setattr(mod, 'NUM_ROUNDS', int(NUM_ROUNDS))
            except Exception:
                pass

            _pool_mod = mod
            return _pool_mod
        except Exception:
            _pool_mod = None
            return None

    def _pool_opponent_names() -> List[str]:
        nonlocal _pool_names_cache
        if _pool_names_cache is not None:
            return list(_pool_names_cache)
        mod = _load_pool_module()
        if mod is None:
            _pool_names_cache = []
            return []
        names: List[str] = []
        try:
            pool_list = getattr(mod, 'OPPONENT_POOL', None)
            if isinstance(pool_list, list):
                for cls in pool_list:
                    try:
                        nm = str(getattr(cls, '__name__', '') or '').strip()
                        if nm:
                            names.append(nm)
                    except Exception:
                        continue
        except Exception:
            names = []
        # Preserve order but unique.
        seen = set()
        uniq: List[str] = []
        for nm in names:
            if nm not in seen:
                seen.add(nm)
                uniq.append(nm)
        _pool_names_cache = list(uniq)
        return list(_pool_names_cache)

    def _new_pool_opponent():
        mod = _load_pool_module()
        if mod is None:
            return None
        BotCls = getattr(mod, 'PoolOpponent', None)
        if BotCls is None:
            return None
        try:
            return BotCls()
        except Exception:
            return None

    def _load_heuristic_bot():
        nonlocal _heuristic_bot
        if _heuristic_bot is not None:
            return _heuristic_bot

        # Avoid loading huge precomputed tables during eval.
        os.environ['GOODBOT_USE_PG'] = '0'
        os.environ['GOODBOT_PRECOMPUTED_EQUITY'] = '__none__'

        # Import the heuristic GoodBot from opponent_pool.
        try:
            import importlib
            import importlib.util

            copy2_path = os.path.join(opp_pool_dir, 'player.py')
            mod = None
            if os.path.isfile(copy2_path):
                spec = importlib.util.spec_from_file_location('pool_player', copy2_path)
                if spec is not None and spec.loader is not None:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

            if mod is None:
                mod = importlib.import_module('player')

            # IMPORTANT: opponent_pool/player.py uses skeleton action classes,
            # but we pass engine.RoundState here.
            try:
                setattr(mod, 'FoldAction', engine.FoldAction)
                setattr(mod, 'CallAction', engine.CallAction)
                setattr(mod, 'CheckAction', engine.CheckAction)
                setattr(mod, 'RaiseAction', engine.RaiseAction)
                setattr(mod, 'DiscardAction', engine.DiscardAction)
                setattr(mod, 'STARTING_STACK', int(engine.STARTING_STACK))
                setattr(mod, 'BIG_BLIND', int(engine.BIG_BLIND))
                setattr(mod, 'SMALL_BLIND', int(engine.SMALL_BLIND))
                setattr(mod, 'NUM_ROUNDS', int(NUM_ROUNDS))
            except Exception:
                pass

            BotCls = getattr(mod, 'GoodBot', None)
            _heuristic_bot = BotCls() if BotCls is not None else None
        except Exception:
            _heuristic_bot = None

        return _heuristic_bot

    def play_match_vs_heuristic(learner_seat: int) -> Tuple[List[float], List[List[Tuple[int, List[float], List[int], List[float], int, float, float]]]]:
        """Play one match vs heuristic GoodBot, recording only learner trajectory."""
        bot = _load_heuristic_bot()
        if bot is None:
            return play_match_mirror()

        rounds = int(args.rounds) if int(args.rounds) > 0 else NUM_ROUNDS
        heuristic_seat = 1 - int(learner_seat)
        bankroll = [0.0, 0.0]
        traj: List[List[Tuple[int, List[float], List[int], List[float], int, float, float]]] = [[], []]
        gs = _EvalGS()

        for _h in range(int(rounds)):
            deck = engine.pkrbot.Deck()
            deck.shuffle()
            hands = [deck.deal(3), deck.deal(3)]
            board = []
            pips = [engine.SMALL_BLIND, engine.BIG_BLIND]
            stacks = [engine.STARTING_STACK - engine.SMALL_BLIND, engine.STARTING_STACK - engine.BIG_BLIND]
            rs = engine.RoundState(0, 0, pips, stacks, hands, deck, board, None)

            # Let the heuristic bot reset per-hand state.
            try:
                gs.bankroll = int(bankroll[int(heuristic_seat)])
                gs.round_num = int(_h) + 1
                bot.handle_new_round(gs, rs, int(heuristic_seat))
            except Exception:
                pass

            while not isinstance(rs, engine.TerminalState):
                active = rs.button % 2
                if int(active) == int(learner_seat):
                    act = _choose_action_with_params(
                        rs,
                        active,
                        learner_params,
                        deterministic=False,
                        record_traj=True,
                        traj_out=traj[int(learner_seat)],
                    )
                else:
                    gs.bankroll = int(bankroll[int(heuristic_seat)])
                    ha = bot.get_action(gs, rs, active)
                    act = _convert_heuristic_action_to_engine(ha)
                rs = rs.proceed(act)

            bankroll[0] += float(rs.deltas[0])
            bankroll[1] += float(rs.deltas[1])

            # Assign learner's hand delta to its last recorded action (if any).
            try:
                ls = int(learner_seat)
                if traj[ls]:
                    k, feats, cands, probs, chosen, frac, r_old, _d0 = traj[ls][-1]
                    traj[ls][-1] = (k, feats, cands, probs, chosen, frac, float(r_old) + float(rs.deltas[ls]) / float(BIG_BLIND), 1.0)
            except Exception:
                pass
            try:
                gs.bankroll = int(bankroll[int(heuristic_seat)])
                bot.handle_round_over(gs, rs, int(heuristic_seat))
            except Exception:
                pass
            gs.round_num += 1

        returns = [0.0, 0.0]
        returns[int(learner_seat)] = float(bankroll[int(learner_seat)]) / float(BIG_BLIND)
        return returns, traj

    def play_match_vs_pool(learner_seat: int) -> Tuple[List[float], List[List[Tuple[int, List[float], List[int], List[float], int, float, float]]]]:
        """Play one match vs opponent_pool PoolOpponent, recording only learner trajectory."""
        
        nonlocal _pool_shuffle_bag, _pool_rr_index, _last_pool_opp_name

        # PoolOpponent reads POOL_SEED/POOL_OPPONENT at __init__, so reseed
        # per episode or a whole run faces one opponent.
        try:
            forced_opp = str(getattr(args, 'pool_opponent', '') or '').strip()
            sampling = str(getattr(args, 'pool_sampling', 'random') or 'random').strip().lower()

            if forced_opp:
                os.environ['POOL_OPPONENT'] = forced_opp
            else:
                if sampling in {'shuffle', 'roundrobin'}:
                    names = _pool_opponent_names()
                    if names:
                        # Bias ManiacBot and BluffHeavyBot to be twice as likely as others
                        weights = []
                        for nm in names:
                            if nm.lower() in {"maniacbot", "bluffheavybot", "goodbot2"}:
                                weights.append(2.0)
                            else:
                                weights.append(1.0)
                        total = sum(weights)
                        probs = [w / total for w in weights]
                        chosen = ''
                        if sampling == 'roundrobin':
                            # Roundrobin: keep order, but repeat Maniac/BluffHeavy
                            expanded = []
                            for nm, w in zip(names, weights):
                                expanded.extend([nm] * int(w))
                            if not expanded:
                                os.environ.pop('POOL_OPPONENT', None)
                            else:
                                chosen = expanded[int(_pool_rr_index) % len(expanded)]
                                _pool_rr_index = int(_pool_rr_index) + 1
                        else:
                            # Weighted random
                            chosen = rng.choices(names, weights=weights, k=1)[0]
                        if chosen:
                            os.environ['POOL_OPPONENT'] = str(chosen)
                        else:
                            os.environ.pop('POOL_OPPONENT', None)
                    else:
                        os.environ.pop('POOL_OPPONENT', None)
                else:
                    os.environ.pop('POOL_OPPONENT', None)

            ps = int(getattr(args, 'pool_seed', -1) or -1)
            if ps >= 0:
                os.environ['POOL_SEED'] = str(ps)
            else:
                # Reseed per episode from trainer RNG (reproducible via --seed).
                os.environ['POOL_SEED'] = str(int(rng.randrange(0, 2**31 - 1)))
        except Exception:
            pass

        bot = _new_pool_opponent()
        if bot is None:
            return play_match_mirror()

        # Record the concrete pool opponent selected/forced for logging.
        try:
            impl = getattr(bot, 'impl', None)
            if impl is not None:
                _last_pool_opp_name = str(type(impl).__name__)
            else:
                _last_pool_opp_name = str(type(bot).__name__)
        except Exception:
            _last_pool_opp_name = ''

        rounds = int(args.rounds) if int(args.rounds) > 0 else NUM_ROUNDS
        pool_seat = 1 - int(learner_seat)
        bankroll = [0.0, 0.0]
        traj: List[List[Tuple[int, List[float], List[int], List[float], int, float, float]]] = [[], []]
        gs = _EvalGS()
        
        for _h in range(int(rounds)):
            deck = engine.pkrbot.Deck()
            deck.shuffle()
            hands = [deck.deal(3), deck.deal(3)]
            board = []
            pips = [engine.SMALL_BLIND, engine.BIG_BLIND]
            stacks = [engine.STARTING_STACK - engine.SMALL_BLIND, engine.STARTING_STACK - engine.BIG_BLIND]
            rs = engine.RoundState(0, 0, pips, stacks, hands, deck, board, None)
            
            try:
                gs.bankroll = int(bankroll[int(pool_seat)])
                gs.round_num = int(_h) + 1
                bot.handle_new_round(gs, rs, int(pool_seat))
            except Exception:
                pass

            while not isinstance(rs, engine.TerminalState):
                active = rs.button % 2
                if int(active) == int(learner_seat):
                    act = _choose_action_with_params(
                        rs,
                        active,
                        learner_params,
                        deterministic=False,
                        record_traj=True,
                        traj_out=traj[int(learner_seat)],
                    )
                else:
                    gs.bankroll = int(bankroll[int(pool_seat)])
                    oa = bot.get_action(gs, rs, active)
                    act = _convert_heuristic_action_to_engine(oa)
                rs = rs.proceed(act)

            bankroll[0] += float(rs.deltas[0])
            bankroll[1] += float(rs.deltas[1])

            # Assign learner's hand delta to its last recorded action (if any).
            try:
                ls = int(learner_seat)
                if traj[ls]:
                    k, feats, cands, probs, chosen, frac, r_old, _d0 = traj[ls][-1]
                    traj[ls][-1] = (k, feats, cands, probs, chosen, frac, float(r_old) + float(rs.deltas[ls]) / float(BIG_BLIND), 1.0)
            except Exception:
                pass
            try:
                gs.bankroll = int(bankroll[int(pool_seat)])
                bot.handle_round_over(gs, rs, int(pool_seat))
            except Exception:
                pass
            gs.round_num += 1
        
        returns = [0.0, 0.0]
        returns[int(learner_seat)] = float(bankroll[int(learner_seat)]) / float(BIG_BLIND)
        return returns, traj

    def eval_vs_heuristic(matches: int) -> float:
        bot = _load_heuristic_bot()
        if bot is None:
            return 0.0

        total_bb = 0.0
        rounds = int(args.rounds) if int(args.rounds) > 0 else NUM_ROUNDS

        for _m in range(int(matches)):
            # Alternate seats to reduce blind/position bias.
            learner_seat = int(_m % 2)
            heuristic_seat = 1 - learner_seat

            bankroll = [0.0, 0.0]
            gs = _EvalGS()

            for _h in range(int(rounds)):
                deck = engine.pkrbot.Deck()
                deck.shuffle()
                hands = [deck.deal(3), deck.deal(3)]
                board = []
                pips = [engine.SMALL_BLIND, engine.BIG_BLIND]
                stacks = [engine.STARTING_STACK - engine.SMALL_BLIND, engine.STARTING_STACK - engine.BIG_BLIND]
                rs = engine.RoundState(0, 0, pips, stacks, hands, deck, board, None)

                try:
                    gs.bankroll = int(bankroll[int(heuristic_seat)])
                    gs.round_num = int(_h) + 1
                    bot.handle_new_round(gs, rs, int(heuristic_seat))
                except Exception:
                    pass

                while not isinstance(rs, engine.TerminalState):
                    active = rs.button % 2
                    if int(active) == int(learner_seat):
                        # Use stochastic action selection for evaluation to avoid
                        # degenerate argmax tie-breaking (e.g., always folding).
                        act = _policy_choose_action(rs, active, deterministic=False)
                    else:
                        gs.bankroll = int(bankroll[int(heuristic_seat)])
                        ha = bot.get_action(gs, rs, active)
                        act = _convert_heuristic_action_to_engine(ha)
                    rs = rs.proceed(act)

                bankroll[0] += float(rs.deltas[0])
                bankroll[1] += float(rs.deltas[1])
                try:
                    gs.bankroll = int(bankroll[int(heuristic_seat)])
                    bot.handle_round_over(gs, rs, int(heuristic_seat))
                except Exception:
                    pass
                gs.round_num += 1

            total_bb += float(bankroll[int(learner_seat)]) / float(BIG_BLIND)

        return total_bb / float(max(1, int(matches)))

    def update_from_episode_reinforce(returns_bb: List[float], traj):
        nonlocal v_b
        if use_numpy:
            nonlocal bv

        returns = [float(returns_bb[0]), float(returns_bb[1])]

        lr = float(args.lr)
        lr_v = float(args.lr_v)
        gamma = float(args.gamma)
        clip_adv = float(args.clip_adv)
        clip_w = float(args.clip_w)

        for player in (0, 1):
            G = returns[player]
            for kind, feats, candidates, probs, chosen, f, _r, _d in reversed(traj[player]):
                # baseline
                v = value(feats)
                adv = (G - v)
                if not math.isfinite(adv):
                    # Skip pathological updates.
                    G *= gamma
                    continue
                if clip_adv > 0:
                    adv = max(-clip_adv, min(clip_adv, adv))

                if use_numpy:
                    # REINFORCE path is kept only as fallback; sizing head update is omitted here.
                    x = np.asarray(feats, dtype=np.float64)
                    candidates_np = np.asarray(candidates, dtype=np.int64)
                    probs_np = np.asarray(probs, dtype=np.float64)

                    hpre, h, logits_main, logits_discard, _raw_a, _raw_b, v = forward(x)

                    dv = (v - G)
                    Wv[:] = Wv - (lr_v * dv) * h
                    bv = float(bv - lr_v * dv)

                    # categorical update
                    chosen_pos = 0
                    for i, a in enumerate(candidates_np.tolist()):
                        if int(a) == int(chosen):
                            chosen_pos = i
                            break
                    grad_logits = (np.eye(len(probs_np), dtype=np.float64)[chosen_pos] - probs_np) * adv

                    if int(kind) == int(StepKind.DISCARD):
                        grad_full = np.zeros((3,), dtype=np.float64)
                        for j, a in enumerate(candidates_np.tolist()):
                            grad_full[int(a)] += float(grad_logits[j])
                        Wp_discard[:] = Wp_discard + lr * np.outer(grad_full, h)
                        bp_discard[:] = bp_discard + lr * grad_full
                        gh = Wp_discard.T @ grad_full
                    else:
                        grad_full = np.zeros((MainActionId.N_ACTIONS,), dtype=np.float64)
                        for j, a in enumerate(candidates_np.tolist()):
                            grad_full[int(a)] += float(grad_logits[j])
                        Wp_main[:] = Wp_main + lr * np.outer(grad_full, h)
                        bp_main[:] = bp_main + lr * grad_full
                        gh = Wp_main.T @ grad_full

                    ghpre = gh * (hpre > 0.0)
                    W1[:] = W1 + lr * np.outer(ghpre, x)
                    b1[:] = b1 + lr * ghpre

                    if clip_w > 0:
                        np.clip(W1, -clip_w, clip_w, out=W1)
                        np.clip(b1, -clip_w, clip_w, out=b1)
                        np.clip(Wp_main, -clip_w, clip_w, out=Wp_main)
                        np.clip(bp_main, -clip_w, clip_w, out=bp_main)
                        np.clip(Wp_discard, -clip_w, clip_w, out=Wp_discard)
                        np.clip(bp_discard, -clip_w, clip_w, out=bp_discard)
                        np.clip(w_beta_a, -clip_w, clip_w, out=w_beta_a)
                        np.clip(w_beta_b, -clip_w, clip_w, out=w_beta_b)
                        np.clip(Wv, -clip_w, clip_w, out=Wv)
                        bv = float(max(-clip_w, min(clip_w, bv)))
                else:
                    # Pure python REINFORCE fallback (no sizing head update).
                    for i in range(feature_dim):
                        v_w[i] += lr_v * adv * feats[i]
                    v_b += lr_v * adv

                    for j, act_id in enumerate(candidates):
                        coeff = (1.0 if int(act_id) == int(chosen) else 0.0) - float(probs[j])
                        if int(kind) == int(StepKind.DISCARD):
                            for k in range(feature_dim):
                                W_discard[int(act_id)][k] += lr * adv * coeff * feats[k]
                            b_discard[int(act_id)] += lr * adv * coeff
                        else:
                            for k in range(feature_dim):
                                W_main[int(act_id)][k] += lr * adv * coeff * feats[k]
                            b_main[int(act_id)] += lr * adv * coeff

                G *= gamma

    def _digamma(x: float) -> float:
        # Simple scalar digamma approximation with recurrence to x>=8.
        x = float(x)
        if x <= 0.0:
            return float('nan')
        r = 0.0
        while x < 8.0:
            r -= 1.0 / x
            x += 1.0
        inv = 1.0 / x
        inv2 = inv * inv
        # asymptotic expansion
        return r + math.log(x) - 0.5 * inv - inv2 * (1.0 / 12.0) + (inv2 * inv2) * (1.0 / 120.0) - (inv2 * inv2 * inv2) * (1.0 / 252.0)

    def _beta_logpdf_and_grads(f: float, alpha: float, beta: float):
        # Returns (logpdf, dlogpdf/dalpha, dlogpdf/dbeta)
        f = float(max(1e-6, min(1.0 - 1e-6, float(f))))
        a = float(max(1e-6, float(alpha)))
        b = float(max(1e-6, float(beta)))
        logpdf = (a - 1.0) * math.log(f) + (b - 1.0) * math.log(1.0 - f) - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
        psi_a = _digamma(a)
        psi_b = _digamma(b)
        psi_ab = _digamma(a + b)
        da = math.log(f) - psi_a + psi_ab
        db = math.log(1.0 - f) - psi_b + psi_ab
        return float(logpdf), float(da), float(db)

    def _policy_probs_and_logp_step(x, step):
        kind, _feats, candidates, _probs_old, chosen, f, _r, _d = step
        hpre, h, logits_main, logits_discard, raw_a, raw_b, v = forward(x)
        cand = np.asarray(candidates, dtype=np.int64)

        if int(kind) == int(StepKind.DISCARD):
            logits = logits_discard[cand]
            probs = _softmax_np(logits)
            chosen_pos = 0
            for i, a in enumerate(cand.tolist()):
                if int(a) == int(chosen):
                    chosen_pos = i
                    break
            p_chosen = float(probs[chosen_pos])
            logp = math.log(max(1e-12, p_chosen))
            return probs, float(logp), float(v)

        logits = logits_main[cand]
        probs = _softmax_np(logits)
        chosen_pos = 0
        for i, a in enumerate(cand.tolist()):
            if int(a) == int(chosen):
                chosen_pos = i
                break
        p_chosen = float(probs[chosen_pos])
        logp = math.log(max(1e-12, p_chosen))
        if int(chosen) == int(MainActionId.RAISE) and math.isfinite(float(f)):
            alpha = _softplus(float(raw_a)) + 1e-3
            beta = _softplus(float(raw_b)) + 1e-3
            lp_beta, _da, _db = _beta_logpdf_and_grads(float(f), float(alpha), float(beta))
            logp += float(lp_beta)
        return probs, float(logp), float(v)

    def update_from_episode_ppo(returns_bb: List[float], traj):
        """PPO with GAE, each player's trajectory treated separately."""
        nonlocal v_b
        nonlocal bv
        nonlocal b_beta_a
        nonlocal b_beta_b

        gamma = float(args.gamma)
        lam = float(args.gae_lambda)
        clip_eps = float(args.clip_eps)
        ent_coef = float(args.ent_coef)
        vf_coef = float(args.vf_coef)
        max_grad_norm = float(args.max_grad_norm)
        clip_adv = float(args.clip_adv)
        clip_w = float(args.clip_w)

        # Flatten both players into one dataset.
        X_list = []
        cand_list = []
        chosen_list = []
        old_logp_list = []
        adv_list = []
        ret_list = []

        for player in (0, 1):
            steps = traj[player]
            if not steps:
                continue
            T = len(steps)
            # Dense per-hand reward: each hand's delta is assigned to the last
            # recorded action in that hand (reward_bb field).
            rewards = np.asarray([float(s[6]) for s in steps], dtype=np.float64)

            X = np.asarray([s[1] for s in steps], dtype=np.float64)

            # Old logp and values
            old_logp = np.zeros((T,), dtype=np.float64)
            values = np.zeros((T,), dtype=np.float64)
            for t in range(T):
                _probs, lp, v = _policy_probs_and_logp_step(X[t], steps[t])
                old_logp[t] = float(lp)
                values[t] = float(v)

            # GAE(lambda), reset at every hand boundary. `done` marks the last
            # action of a hand, where its delta was booked. Hands are
            # independent -- stacks reset, cards redealt -- so bootstrapping or
            # carrying lastgaelam across a boundary credits one hand's outcome
            # to the previous hand's actions.
            dones = np.asarray([float(s[7]) for s in steps], dtype=np.float64)
            adv = np.zeros((T,), dtype=np.float64)
            lastgaelam = 0.0
            for t in reversed(range(T)):
                nonterminal = 1.0 - dones[t]
                next_value = 0.0 if t == T - 1 else values[t + 1] * nonterminal
                delta = rewards[t] + gamma * next_value - values[t]
                lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
                adv[t] = lastgaelam
            returns = adv + values

            # Advantage normalization helps stability
            adv_mean = float(np.mean(adv))
            adv_std = float(np.std(adv))
            if adv_std > 1e-6:
                adv = (adv - adv_mean) / adv_std

            if clip_adv > 0:
                adv = np.clip(adv, -clip_adv, clip_adv)

            for t in range(T):
                X_list.append(X[t])
                cand_list.append(steps[t][2])
                chosen_list.append(int(steps[t][4]))
                old_logp_list.append(float(old_logp[t]))
                adv_list.append(float(adv[t]))
                ret_list.append(float(returns[t]))
        
        # Rebuild packed arrays including kind/f for each transition
        kinds = []
        fracs = []
        for player in (0, 1):
            for step in traj[player]:
                kinds.append(int(step[0]))
                fracs.append(float(step[5]))
            # kinds/fracs follow the concatenation order used above.

        if not X_list:
            return

        N = len(X_list)
        if len(kinds) != N or len(fracs) != N:
            # Safety: if alignment is off, skip update.
            return
        idxs = np.arange(N)
        np_rng.shuffle(idxs)

        lr = float(args.lr)
        lr_v = float(args.lr_v)

        for _epoch in range(int(args.ppo_epochs)):
            np_rng.shuffle(idxs)
            for start in range(0, N, int(args.minibatch)):
                mb = idxs[start : start + int(args.minibatch)]
                # Accumulate grads
                gW1 = np.zeros_like(W1)
                gb1 = np.zeros_like(b1)
                gWp_main = np.zeros_like(Wp_main)
                gbp_main = np.zeros_like(bp_main)
                gWp_discard = np.zeros_like(Wp_discard)
                gbp_discard = np.zeros_like(bp_discard)
                gw_beta_a = np.zeros_like(w_beta_a)
                gb_beta_a = 0.0
                gw_beta_b = np.zeros_like(w_beta_b)
                gb_beta_b = 0.0
                gWv = np.zeros_like(Wv)
                gbv = 0.0

                for i in mb.tolist():
                    x = X_list[i]
                    candidates = cand_list[i]
                    chosen = int(chosen_list[i])
                    old_logp = float(old_logp_list[i])
                    adv = float(adv_list[i])
                    ret = float(ret_list[i])
                    kind = int(kinds[i])
                    f = float(fracs[i])

                    # current probs/logp
                    hpre, h, logits_main, logits_discard, raw_a, raw_b, v = forward(x)

                    cand = np.asarray(candidates, dtype=np.int64)
                    if kind == StepKind.DISCARD:
                        logits = logits_discard[cand]
                    else:
                        logits = logits_main[cand]

                    probs = _softmax_np(logits)
                    # chosen position
                    chosen_pos = 0
                    for j, a in enumerate(cand.tolist()):
                        if int(a) == int(chosen):
                            chosen_pos = j
                            break
                    p_chosen = float(probs[chosen_pos])
                    logp = math.log(max(1e-12, p_chosen))

                    # Add continuous logp for parameterized raise.
                    dlogp_draw_a = 0.0
                    dlogp_draw_b = 0.0
                    if kind == StepKind.MAIN and int(chosen) == int(MainActionId.RAISE) and math.isfinite(float(f)):
                        alpha = _softplus(float(raw_a)) + 1e-3
                        beta = _softplus(float(raw_b)) + 1e-3
                        lp_beta, da, db = _beta_logpdf_and_grads(float(f), float(alpha), float(beta))
                        logp += float(lp_beta)
                        # chain through softplus
                        dlogp_draw_a = float(da) * float(_sigmoid(float(raw_a)))
                        dlogp_draw_b = float(db) * float(_sigmoid(float(raw_b)))

                    ratio = math.exp(logp - old_logp)
                    clipped = max(1.0 - clip_eps, min(1.0 + clip_eps, ratio))
                    s1 = ratio * adv
                    s2 = clipped * adv
                    surr = s1 if s1 < s2 else s2

                    # dL_policy/dlogp
                    if surr == s1:
                        dL_dlogp = -adv * ratio
                    else:
                        # clipped region -> zero gradient w.r.t logp
                        dL_dlogp = 0.0

                    # policy gradient wrt logits via dlogp/dlogits = onehot - probs
                    grad_logits = dL_dlogp * (np.eye(len(probs), dtype=np.float64)[chosen_pos] - probs)

                    # entropy bonus gradient
                    if ent_coef != 0.0:
                        # dH/dz_k = p_k * (sum_j p_j log p_j - log p_k)
                        logp_all = np.log(np.maximum(1e-12, probs))
                        sum_p_logp = float(np.sum(probs * logp_all))
                        dH_dz = probs * (sum_p_logp - logp_all)
                        grad_logits = grad_logits - ent_coef * dH_dz

                    # Map candidate logits grad -> full head grad
                    if kind == StepKind.DISCARD:
                        grad_logits_all = np.zeros((3,), dtype=np.float64)
                    else:
                        grad_logits_all = np.zeros((MainActionId.N_ACTIONS,), dtype=np.float64)
                    for j, a in enumerate(cand.tolist()):
                        grad_logits_all[int(a)] += float(grad_logits[j])

                    # Policy head
                    if kind == StepKind.DISCARD:
                        gWp_discard += np.outer(grad_logits_all, h)
                        gbp_discard += grad_logits_all
                        gh = Wp_discard.T @ grad_logits_all
                    else:
                        gWp_main += np.outer(grad_logits_all, h)
                        gbp_main += grad_logits_all
                        gh = Wp_main.T @ grad_logits_all

                    # Beta head gradients (only when the chosen action was RAISE)
                    if kind == StepKind.MAIN and int(chosen) == int(MainActionId.RAISE) and dL_dlogp != 0.0:
                        draw_a = float(dL_dlogp) * float(dlogp_draw_a)
                        draw_b = float(dL_dlogp) * float(dlogp_draw_b)
                        gw_beta_a += draw_a * h
                        gb_beta_a += float(draw_a)
                        gw_beta_b += draw_b * h
                        gb_beta_b += float(draw_b)
                        gh = gh + (draw_a * w_beta_a) + (draw_b * w_beta_b)

                    # Value head
                    dv = vf_coef * (v - ret)
                    gWv += dv * h
                    gbv += float(dv)
                    gh = gh + (dv * Wv)

                    # Backprop through ReLU
                    ghpre = gh * (hpre > 0.0)
                    gW1 += np.outer(ghpre, x)
                    gb1 += ghpre

                # average grads
                mbn = float(max(1, len(mb)))
                gW1 /= mbn
                gb1 /= mbn
                gWp_main /= mbn
                gbp_main /= mbn
                gWp_discard /= mbn
                gbp_discard /= mbn
                gw_beta_a /= mbn
                gb_beta_a /= mbn
                gw_beta_b /= mbn
                gb_beta_b /= mbn
                gWv /= mbn
                gbv /= mbn

                # global grad norm clip
                if max_grad_norm and max_grad_norm > 0:
                    gn = float(
                        np.sqrt(
                            np.sum(gW1 * gW1)
                            + np.sum(gb1 * gb1)
                            + np.sum(gWp_main * gWp_main)
                            + np.sum(gbp_main * gbp_main)
                            + np.sum(gWp_discard * gWp_discard)
                            + np.sum(gbp_discard * gbp_discard)
                            + np.sum(gw_beta_a * gw_beta_a)
                            + float(gb_beta_a * gb_beta_a)
                            + np.sum(gw_beta_b * gw_beta_b)
                            + float(gb_beta_b * gb_beta_b)
                            + np.sum(gWv * gWv)
                            + float(gbv * gbv)
                        )
                    )
                    if gn > max_grad_norm:
                        scale = max_grad_norm / max(1e-12, gn)
                        gW1 *= scale
                        gb1 *= scale
                        gWp_main *= scale
                        gbp_main *= scale
                        gWp_discard *= scale
                        gbp_discard *= scale
                        gw_beta_a *= scale
                        gb_beta_a *= scale
                        gw_beta_b *= scale
                        gb_beta_b *= scale
                        gWv *= scale
                        gbv *= scale

                # SGD step
                W1[:] = W1 - lr * gW1
                b1[:] = b1 - lr * gb1
                Wp_main[:] = Wp_main - lr * gWp_main
                bp_main[:] = bp_main - lr * gbp_main
                Wp_discard[:] = Wp_discard - lr * gWp_discard
                bp_discard[:] = bp_discard - lr * gbp_discard
                w_beta_a[:] = w_beta_a - lr * gw_beta_a
                b_beta_a = float(b_beta_a - lr * gb_beta_a)
                w_beta_b[:] = w_beta_b - lr * gw_beta_b
                b_beta_b = float(b_beta_b - lr * gb_beta_b)
                Wv[:] = Wv - lr_v * gWv
                bv = float(bv - lr_v * gbv)

                if clip_w > 0:
                    np.clip(W1, -clip_w, clip_w, out=W1)
                    np.clip(b1, -clip_w, clip_w, out=b1)
                    np.clip(Wp_main, -clip_w, clip_w, out=Wp_main)
                    np.clip(bp_main, -clip_w, clip_w, out=bp_main)
                    np.clip(Wp_discard, -clip_w, clip_w, out=Wp_discard)
                    np.clip(bp_discard, -clip_w, clip_w, out=bp_discard)
                    np.clip(w_beta_a, -clip_w, clip_w, out=w_beta_a)
                    np.clip(w_beta_b, -clip_w, clip_w, out=w_beta_b)
                    b_beta_a = float(max(-clip_w, min(clip_w, b_beta_a)))
                    b_beta_b = float(max(-clip_w, min(clip_w, b_beta_b)))
                    np.clip(Wv, -clip_w, clip_w, out=Wv)
                    bv = float(max(-clip_w, min(clip_w, bv)))

    # --- League init ---
    league_dir = os.path.join(root, args.league_dir) if not os.path.isabs(args.league_dir) else args.league_dir
    os.makedirs(league_dir, exist_ok=True)
    if use_numpy:
        snaps = _list_league_snapshots(league_dir)
        if not snaps:
            pth = _freeze_to_league(league_dir, 0)
            print(f'league: initialized with {os.path.basename(pth)}')
        _prune_league(league_dir, int(args.max_league_size))

    # --- Best checkpoint tracking ---
    out_path_base = os.path.join(root, args.out) if not os.path.isabs(args.out) else args.out
    save_best = bool(getattr(args, 'save_best', False))
    best_metric = str(getattr(args, 'best_metric', 'eval') or 'eval').strip().lower()
    best_out = str(getattr(args, 'best_out', '') or '').strip()
    if save_best and not best_out:
        suffix = 'best_eval' if best_metric == 'eval' else 'best_ema'
        best_out = str(out_path_base) + f'.{suffix}.json'

    # Restore previous best score if present so we don't overwrite best_out with a worse model.
    best_score = -1e30
    if save_best and best_out:
        try:
            if os.path.isfile(best_out):
                with open(best_out, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                meta = data.get('meta', {}) if isinstance(data, dict) else {}
                if isinstance(meta, dict):
                    bs = meta.get('best_score', None)
                    bm = str(meta.get('best_metric', '') or '').strip().lower()
                    if isinstance(bs, (int, float)) and math.isfinite(float(bs)) and (not bm or bm == best_metric):
                        best_score = float(bs)
        except Exception:
            pass

    # Initialize learner from a checkpoint (default: best_out if it exists).
    init_from = str(getattr(args, 'init_from', '') or '').strip()
    init_from_best = bool(getattr(args, 'init_from_best', True))
    if use_numpy:
        if init_from:
            _try_init_learner_from(init_from)
        elif init_from_best and best_out and os.path.isfile(best_out):
            _try_init_learner_from(best_out)

    avg = 0.0
    for ep in range(1, int(args.episodes) + 1):
        learner_seat = 0
        opp_path = None
        if use_numpy:
            snaps = _list_league_snapshots(league_dir)
            if not snaps:
                pth = _freeze_to_league(league_dir, 0)
                snaps = [pth]

            if bool(getattr(args, 'randomize_seat', False)):
                learner_seat = int(rng.choice([0, 1]))
            else:
                learner_seat = 0

            p_vs_pool = float(getattr(args, 'train_vs_pool_prob', 0.0) or 0.0)
            p_vs_heur = float(getattr(args, 'train_vs_heuristic_prob', 0.0) or 0.0)
            total_prob = p_vs_pool + p_vs_heur
            r = rng.random()
            if total_prob >= 1.0:
                if r < p_vs_pool / total_prob:
                    returns_bb, traj = play_match_vs_pool(learner_seat)
                    opp_path = 'opponent_pool'
                else:
                    returns_bb, traj = play_match_vs_heuristic(learner_seat)
                    opp_path = 'heuristic_goodbot'
            else:
                if p_vs_pool > 0.0 and r < p_vs_pool:
                    returns_bb, traj = play_match_vs_pool(learner_seat)
                    opp_path = 'opponent_pool'
                elif p_vs_heur > 0.0 and r < p_vs_pool + p_vs_heur:
                    returns_bb, traj = play_match_vs_heuristic(learner_seat)
                    opp_path = 'heuristic_goodbot'
                else:
                    opp_path = rng.choice(snaps)
                    try:
                        opponent_params = _load_mlp_params_from_json(opp_path, exact_dims=False)
                    except Exception:
                        opponent_params = _load_mlp_params_from_json(snaps[-1], exact_dims=False)
                    returns_bb, traj = play_match_vs_snapshot(opponent_params, learner_seat)
        else:
            # Non-numpy fallback: preserve old mirror self-play behavior.
            returns_bb, traj = play_match_mirror()

        if args.algo == 'ppo':
            update_from_episode_ppo(returns_bb, traj)
        else:
            update_from_episode_reinforce(returns_bb, traj)

        if ep < 10 or ep % 100 == 0:  # First 10 episodes, then every 100
            print(f"\n=== EPISODE {ep} DEBUG ===")
            print(f"returns_bb = {returns_bb}")
            print(f"learner_seat = {learner_seat}")
            
            total_actions = sum(len(traj[s]) for s in (0, 1))
            total_reward = sum(step[6] for s in (0, 1) for step in traj[s])
            
            print(f"Total actions in match: {total_actions}")
            print(f"Total reward in match: {total_reward:.2f}")
            
            all_rewards = [step[6] for s in (0, 1) for step in traj[s]]
            if all_rewards:
                min_r = min(all_rewards)
                max_r = max(all_rewards)
                print(f"Min reward: {min_r:.2f}, Max reward: {max_r:.2f}")
                
                if abs(total_reward) > 1000:
                    print("[warn] large accumulated reward in this match; tracing first steps. "
                          "Note each episode is a full match, so a lopsided result can "
                          "legitimately exceed the 1000 threshold.")
                    for i, step in enumerate(traj[learner_seat][:10]):
                        print(f"  Action {i}: reward={step[6]:.2f}")
        r_learner = float(returns_bb[int(learner_seat)])
        avg = 0.99 * avg + 0.01 * r_learner

        if save_best and best_metric == 'ema':
            if float(avg) > float(best_score):
                best_score = float(avg)
                payload = _payload_from_current(meta={'best_metric': 'ema', 'best_score': float(best_score), 'episode': int(ep)})
                _atomic_write_json(best_out, payload)
                print(f'best(ema): ep={ep} ema_learner_bb={best_score:.4f} wrote={best_out}')
        if args.log_every and ep % int(args.log_every) == 0:
            if str(opp_path) == 'heuristic_goodbot':
                opp_name = 'heuristic_goodbot'
            elif str(opp_path) == 'opponent_pool':
                opp_name = 'opponent_pool'
                try:
                    if _last_pool_opp_name:
                        opp_name = f'opponent_pool::{_last_pool_opp_name}'
                except Exception:
                    pass
            else:
                opp_name = os.path.basename(str(opp_path)) if opp_path else 'mirror'
            print(f'ep={ep} learner_seat={int(learner_seat)} learner_bb={r_learner:.4f} ema_learner_bb={avg:.4f} opp={opp_name}')

        if args.eval_every and int(args.eval_every) > 0 and (ep % int(args.eval_every) == 0):
            bb = eval_vs_heuristic(int(args.eval_matches))
            print(f'ep={ep} eval_vs_heuristic_goodbot_avg_bb={bb:.4f}')

            if save_best and best_metric == 'eval':
                if float(bb) > float(best_score):
                    best_score = float(bb)
                    payload = _payload_from_current(meta={'best_metric': 'eval', 'best_score': float(best_score), 'episode': int(ep)})
                    _atomic_write_json(best_out, payload)
                    print(f'best(eval): ep={ep} eval_bb={best_score:.4f} wrote={best_out}')

        if use_numpy and args.freeze_every and int(args.freeze_every) > 0 and (ep % int(args.freeze_every) == 0):
            pth = _freeze_to_league(league_dir, int(ep))
            _prune_league(league_dir, int(args.max_league_size))
            print(f'league: froze snapshot {os.path.basename(pth)}')

    if ep % 100 == 0:
        entropy_sum = 0.0
        action_count = 0
        for seat in (0, 1):
            for step in traj[seat]:
                probs = step[3]  # Action probabilities
                entropy = -sum(p * math.log(max(p, 1e-10)) for p in probs)
                entropy_sum += entropy
                action_count += 1
        
        avg_entropy = entropy_sum / max(1, action_count)

        value_error = 0.0
        for seat in (0, 1):
            steps = traj[seat]
            if len(steps) > 0:
                last_value = value(steps[-1][1])  # V(s_last)
                actual_return = returns_bb[seat]
                value_error += abs(last_value - actual_return)
        
        print(f"Metrics - Entropy: {avg_entropy:.3f}, Value Error: {value_error:.3f}, "
              f"Actions/Hand: {action_count/args.rounds:.2f}")
    out_path = os.path.join(root, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    payload = _payload_from_current()

    _atomic_write_json(out_path, payload)

    print(f'Wrote policy to: {out_path}')
    if use_numpy:
        print('Used NumPy fast path.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())