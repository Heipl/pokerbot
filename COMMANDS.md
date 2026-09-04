# mccfr_bot

train
```bash
python tools/train_mccfr.py --traversals 600000
python tools/train_goodbot_pg.py --episodes 50000
```

play
```bash
# config.py: PLAYER_1_PATH = (_HERE / "mccfr_bot").as_posix()
python engine.py
GOODBOT_USE_PG=1 python engine.py   # with the PG policy
```

# neural_cfr_bot

train
```bash
python tools/train_hand_strength.py --samples 200000
```

play
```bash
# config.py: PLAYER_1_PATH = (_HERE / "neural_cfr_bot").as_posix()
python engine.py
```

# opponent_pool

train
```bash
python tools/precompute_goodbot_equity.py --mode preflop_exact
python tools/precompute_goodbot_equity.py --mode texture
python tools/train_goodbot_params.py --iters 25
```

play
```bash
# config.py: PLAYER_2_PATH = (_HERE / "opponent_pool").as_posix()
POOL_OPPONENT=GoodBot python engine.py
```
