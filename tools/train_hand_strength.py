#!/usr/bin/env python3
"""Train HandStrengthNetwork by distilling slow Monte-Carlo equity.

Postflop, pkrbot.evaluate is exact and fast, so neural_cfr_bot just calls it.
Preflop the hand is THREE cards and its value is the option value of choosing
which two to keep once more board arrives -- no closed form covers that, and a
2-card proxy is blind to it (Chen scores AKs, AKQs and AKs+brick identically).
Simulating the three keeps does capture it but is far too slow per decision, so
compute it offline and fit the network to it.

  python tools/train_hand_strength.py --samples 200000
  python tools/train_hand_strength.py --samples 20000 --epochs 6   # quick check

Writes neural_cfr_bot/complete_neural_cfr_models/hand_strength.pt, which
neural_cfr_bot loads at startup if present. Target generation dominates the
runtime and is pure CPU.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pkrbot  # noqa: E402

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['c', 'd', 'h', 's']
# Must match neural_cfr_bot._create_card_mapping exactly, or the embedding
# indices mean different cards at training time and at inference time.
CARD_TO_INDEX = {r + s: i for i, s in enumerate(SUITS) for r in RANKS
                 for i in [len(RANKS) * SUITS.index(s) + RANKS.index(r)]}
DECK = [r + s for s in SUITS for r in RANKS]

MAX_HOLE_CARDS = 3
MAX_BOARD_CARDS = 6
MAX_TOTAL_CARDS = MAX_HOLE_CARDS + MAX_BOARD_CARDS


def best_keep_equity(hole3, rollouts, rng):
    """Equity of the best 2-card keep, against a uniform villain range.

    The max over keeps, not the average: we choose after seeing more board.
    """
    keeps = [(hole3[0], hole3[1]), (hole3[0], hole3[2]), (hole3[1], hole3[2])]
    remaining = [c for c in DECK if c not in hole3]
    best = 0.0
    per_keep = max(1, rollouts // 3)
    for keep in keeps:
        won = 0.0
        for _ in range(per_keep):
            pool = remaining[:]
            rng.shuffle(pool)
            villain = pool[:2]
            board = pool[2:8]
            try:
                mine = int(pkrbot.evaluate([pkrbot.Card(c) for c in board + list(keep)]))
                theirs = int(pkrbot.evaluate([pkrbot.Card(c) for c in board + villain]))
            except Exception:
                continue
            won += 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
        best = max(best, won / per_keep)
    return best


def encode(hole, board):
    """Same layout _estimate_hand_strength builds: hole in slots 0..2, board 3..8."""
    idx, pos = [], []
    for i in range(MAX_HOLE_CARDS):
        if i < len(hole):
            idx.append(CARD_TO_INDEX[hole[i]]); pos.append(i)
        else:
            idx.append(0); pos.append(0)
    for i in range(MAX_BOARD_CARDS):
        if i < len(board):
            idx.append(CARD_TO_INDEX[board[i]]); pos.append(MAX_HOLE_CARDS + i)
        else:
            idx.append(0); pos.append(0)
    return idx, pos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--samples', type=int, default=200000)
    # Signal variance across random 3-card hands is ~0.0115; target noise is
    # ~0.24/R per keep. Below ~210 rollouts per keep the net fits coin flips.
    ap.add_argument('--rollouts', type=int, default=630,
                    help='rollouts per sample, split over the 3 keeps '
                         '(needs ~630 for target noise to be ~10%% of signal)')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=os.path.join(ROOT, 'neural_cfr_bot',
                                                  'complete_neural_cfr_models',
                                                  'hand_strength.pt'))
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    sys.path.insert(0, os.path.join(ROOT, 'neural_cfr_bot'))
    os.environ.setdefault('NCFR_TRAINING', '0')
    from player import HandStrengthNetwork

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)

    # ---------- targets ----------
    print(f"generating {a.samples:,} samples at {a.rollouts} rollouts each...")
    t0 = time.time()
    X_idx, X_pos, Y = [], [], []
    for n in range(a.samples):
        hole = rng.sample(DECK, 3)
        y = best_keep_equity(hole, a.rollouts, rng)
        i, p = encode(hole, [])
        X_idx.append(i); X_pos.append(p); Y.append(y)
        if (n + 1) % max(1, a.samples // 20) == 0:
            el = time.time() - t0
            print(f"  {n+1:>8,}/{a.samples:,}  {el:6.0f}s  eta {el/(n+1)*(a.samples-n-1):6.0f}s",
                  flush=True)
    print(f"targets done in {time.time()-t0:.0f}s; "
          f"mean {sum(Y)/len(Y):.3f}, min {min(Y):.3f}, max {max(Y):.3f}")

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xi = torch.tensor(X_idx, dtype=torch.long, device=dev)
    Xp = torch.tensor(X_pos, dtype=torch.long, device=dev)
    Yt = torch.tensor(Y, dtype=torch.float32, device=dev).unsqueeze(1)

    n_val = max(1, len(Y) // 10)
    tr = slice(0, len(Y) - n_val)
    va = slice(len(Y) - n_val, len(Y))

    net = HandStrengthNetwork().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
    lossf = nn.MSELoss()

    # If the net cannot beat "predict the mean" it has learned nothing.
    base = float(((Yt[va] - Yt[tr].mean()) ** 2).mean())
    print(f"\ndevice {dev} | train {len(Y)-n_val:,} val {n_val:,}")
    print(f"baseline val MSE (predict the mean): {base:.5f}\n")

    n_tr = len(Y) - n_val
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(n_tr, device=dev)
        tot = 0.0
        for s in range(0, n_tr, a.batch):
            b = perm[s:s + a.batch]
            opt.zero_grad(set_to_none=True)
            out = net(Xi[tr][b], Xp[tr][b])
            loss = lossf(out, Yt[tr][b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(b)
        net.eval()
        with torch.no_grad():
            vp = net(Xi[va], Xp[va])
            vl = float(lossf(vp, Yt[va]))
            corr = float(torch.corrcoef(torch.stack([vp.squeeze(1), Yt[va].squeeze(1)]))[0, 1])
        print(f"epoch {ep+1:>3}  train {tot/n_tr:.5f}  val {vl:.5f}  "
              f"r={corr:.4f}  ({100*(1-vl/base):+.1f}% vs baseline)")

    if vl >= base:
        print("\nERROR: the network does not beat predicting the mean. Not saving.")
        return 1

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({'state_dict': net.state_dict(), 'val_mse': vl,
                'baseline_mse': base, 'samples': a.samples,
                'rollouts': a.rollouts}, a.out)
    print(f"\nsaved {a.out}")
    print("neural_cfr_bot loads this at startup if present.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
