"""Regression tests: one defect per test, asserting only the invariant that was
violated, so a reintroduction fails loudly and cheaply.

  python -m pytest tools/test_regressions.py -q
"""
import ast as _ast
import importlib.util
import io
import itertools
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mccfr_bot"))


def _src(rel):
    return io.open(ROOT / rel, encoding="utf-8").read()


def _load_tuner():
    if "tgp" in sys.modules:
        return sys.modules["tgp"]
    spec = importlib.util.spec_from_file_location(
        "tgp", ROOT / "tools/train_goodbot_params.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tgp"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- train_mccfr

def test_bool_env_does_not_treat_zero_as_true():
    """bool('0') is True in Python; an explicit off must stay off."""
    code = (
        "import sys,importlib.util as u;"
        "spec=u.spec_from_file_location('m',r'{}');"
        "m=u.module_from_spec(spec);sys.modules['m']=m;spec.loader.exec_module(m);"
        "print('ON' if m._EXTERNAL_SAMPLING else 'OFF')"
    ).format(ROOT / "tools/train_mccfr.py")
    cases = [(None, "OFF"), ("0", "OFF"), ("false", "OFF"), ("off", "OFF"),
             ("no", "OFF"), ("1", "ON"), ("yes", "ON"), ("true", "ON")]
    for val, want in cases:
        env = dict(os.environ)
        env.pop("MCCFR_EXTERNAL_SAMPLING", None)
        if val is not None:
            env["MCCFR_EXTERNAL_SAMPLING"] = val
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env, cwd=str(ROOT))
        got = [l for l in out.stdout.splitlines() if l in ("ON", "OFF")]
        assert got and got[-1] == want, "{!r} -> {} (want {})".format(val, got, want)


def test_discard_action_field_is_card_not_card_index():
    """getattr(act,'card_index') always missed, collapsing every discard to DISCARD_0."""
    from skeleton.actions import DiscardAction
    src = _src("tools/train_mccfr.py")
    assert 'getattr(act, "card_index"' not in src
    assert 'getattr(act, "card", 0)' in src
    for i in (0, 1, 2):
        assert getattr(DiscardAction(i), "card", None) == i


def test_pool_adapter_snaps_only_to_legal_raise_buckets():
    """Fraction keys must be filtered by the legal mask or raises silently fall back."""
    from mccfr_common import MCCFR_ALLOWED_RAISES, MCCFR_RAISE_FRACTIONS
    src = _src("tools/train_mccfr.py")
    assert "legal_mask[a_idx] <= 0" in src, "legality filter missing from bucket snapping"
    allowed = set(MCCFR_ALLOWED_RAISES)
    for street, fm in MCCFR_RAISE_FRACTIONS.items():
        usable = [k for k in fm if k in allowed]
        assert usable, "street {} has no legal snap target".format(street)
        for frac in (0.3, 0.5, 1.0, 1.5, 2.0, 2.5):
            assert min(usable, key=lambda k: abs(fm[k] - frac)) in allowed


def test_pool_adapter_frac_inverts_proceed_exactly():
    """bucket -> amount -> frac must recover the same bucket, facing a bet too.

    The inverse has to keep cc in both halves or it is wrong for every re-raise.
    """
    from mccfr_common import MCCFR_ALLOWED_RAISES, MCCFR_RAISE_FRACTIONS
    src = _src("tools/train_mccfr.py")
    assert "float(state.pot()) + float(cc)" in src, "cc missing from the denominator"

    fm = MCCFR_RAISE_FRACTIONS["default"]
    usable = [k for k in fm if k in set(MCCFR_ALLOWED_RAISES)]
    for cc in (0, 20, 50):
        for pot in (30, 60, 120):
            for pips in (0, 10, 20):
                for bucket in usable:
                    amount = int(pips + cc + max(1.0, pot + cc) * fm[bucket])
                    frac = max(0.0, amount - pips - cc) / max(1.0, pot + cc)
                    got = min(usable, key=lambda k: abs(fm[k] - frac))
                    assert got == bucket, (cc, pot, pips, bucket, got)


def test_new_hand_called_per_traversal_not_per_pair():
    """Each traversal deals a fresh hand, so the pool bot must be reset each time."""
    assert "opp.new_hand(trainer.traversals + 1 + player)" in _src("tools/train_mccfr.py")


def test_raise_allin_constant_defined_once():
    assert _src("tools/train_mccfr.py").count("\nRAISE_ALLIN = 7\n") == 1


# ------------------------------------------------------------- opponent_pool

def test_goodbot_default_sizings_match_original_hardcoded_values():
    """_p()'s contract: the default must be the value that was hardcoded."""
    src = _src("opponent_pool/player.py")
    assert "_p('facing_bet_raise_pot_hi', 1.0)" in src
    assert "_p('unchecked_bet_pot_hi', 0.75)" in src
    hi, lo = 1.0, 0.5
    assert [hi, (lo + hi) / 2.0, lo] == [1.0, 0.75, 0.5]


def test_raise_size_knob_is_not_capped_at_one():
    """min(_hi,1.0) made the tuner's whole >1.0 range behaviourally dead."""
    assert "min(_hi, 1.0)" not in _src("opponent_pool/player.py")


def test_p_fold_is_clamped_to_probability_range():
    """p_fold is used as (1 - p_fold); two independently clamped knobs can sum past 1."""
    assert "p_fold = min(1.0, max(0.0, p_fold))" in _src("opponent_pool/player.py")
    for base, mult, eq in [(1.0, 2.0, 1.0), (0.0, 0.4, 1.0), (0.5, 0.5, 0.5)]:
        assert 0.0 <= min(1.0, max(0.0, base + mult * eq)) <= 1.0


# --------------------------------------------------------- train_goodbot_params

def test_param_tuner_scores_the_seat_opponent_pool_occupies():
    """Scoring player1 while GoodBot sits at player2 inverts the objective."""
    m = _load_tuner()
    from config import PLAYER_1_NAME, PLAYER_2_NAME, PLAYER_1_PATH, PLAYER_2_PATH
    assert "opponent_pool" in (str(PLAYER_1_PATH) + str(PLAYER_2_PATH))
    expected = PLAYER_1_NAME if "opponent_pool" in str(PLAYER_1_PATH) else PLAYER_2_NAME
    assert m.SCORED_NAME == expected


def test_tuner_params_defaults_reproduce_goodbot():
    """A default params file must not change the reference bot."""
    d = _load_tuner().Params().to_dict()
    assert d["facing_bet_raise_pot_hi"] == 1.0
    assert d["unchecked_bet_pot_hi"] == 0.75


def test_equity_knobs_are_actually_mutated():
    """Params + clamp but not float_keys => the knob never moves and its guard is dead."""
    keys_block = _src("tools/train_goodbot_params.py").split("float_keys = [")[1].split("]")[0]
    assert '"raise_size_equity_hi"' in keys_block
    assert '"raise_size_equity_mid"' in keys_block


# ------------------------------------------------------------- pg / features

def test_appended_features_keep_original_indices():
    """A narrower policy must get exactly the prefix it was trained on."""
    import pg_features

    class RS:
        hands = [["Ah", "Kd", "2c"], ["3h", "4d", "5c"]]
        board = ["7h", "8d", "9c"]
        street, pips, stacks, button = 4, [2, 2], [398, 398], 0

        def raise_bounds(self):
            return (4, 398)

    full = pg_features.extract_features(RS(), 0, 400, 2, None)
    assert len(full) == pg_features.feature_size()
    narrow = pg_features.extract_features(RS(), 0, 400, 2, 184)
    assert len(narrow) == 184
    assert list(full[:184]) == list(narrow)


def test_pg_policy_guard_accepts_narrower_policy():
    src = _src("mccfr_bot/player.py")
    assert "feature_dim > int(pg_features.feature_size())" in src
    assert "feature_dim != int(pg_features.feature_size())" not in src


def test_league_loader_supports_mixed_width_snapshots():
    src = _src("tools/train_goodbot_pg.py")
    assert "exact_dims" in src
    assert "exp_dim" in src and "exp_hid" in src
    assert "dim=int(getattr(params, 'feature_dim', 0) or 0)" in src


def test_trainer_puts_bot_dir_on_syspath():
    """Without it pg_features' mccfr_common import fails and 4 features pin to zero."""
    assert "sys.path.insert(0, bot_dir)" in _src("tools/train_goodbot_pg.py")


def test_gae_masks_hand_boundaries():
    src = _src("tools/train_goodbot_pg.py")
    assert "dones = np.asarray([float(s[7]) for s in steps]" in src
    assert "nonterminal = 1.0 - dones[t]" in src
    assert "gamma * lam * nonterminal * lastgaelam" in src


def test_three_card_hole_scores_best_keepable_two():
    """The player discards to two; scoring all three rates an unreachable hand."""
    for rel in ("mccfr_bot/pg_features.py", "neural_cfr_bot/player.py"):
        assert "combinations(_h, 2)" in _src(rel), rel


def test_vendored_chen_matches_original():
    """neural_cfr_bot must not import across bot directories, but must agree."""
    from mccfr_common import _chen_score as orig
    src = _src("neural_cfr_bot/player.py")
    tree = _ast.parse(src)
    # Check real import statements, not the phrase as it appears in comments.
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ImportFrom):
            assert node.module != "mccfr_common", "cross-bot import reintroduced"
    keep = [n for n in tree.body
            if (isinstance(n, _ast.FunctionDef) and n.name == "_chen_score_local")
            or (isinstance(n, _ast.Assign)
                and any(getattr(t, "id", "").startswith("_CHEN") for t in n.targets))]
    ns = {}
    exec(compile(_ast.Module(body=keep, type_ignores=[]), "<vendored>", "exec"), ns)
    new = ns["_chen_score_local"]
    for r1, r2 in itertools.product("23456789TJQKA", repeat=2):
        for s1, s2 in (("c", "c"), ("c", "d")):
            if r1 == r2 and s1 == s2:
                continue
            a, b = r1 + s1, r2 + s2
            assert abs(orig(a, b) - new(a, b)) < 1e-9, (a, b)


# ------------------------------------------------------------ neural_cfr_bot

def test_gradient_accumulation_is_per_head():
    """A shared counter starved policy/combiner/avg-strategy of every step."""
    src = _src("neural_cfr_bot/player.py")
    assert "_accum_ready" in src
    # 'critic' and 'encoder' are shared modules stepped once by _train_networks;
    # the MCCFR value loss only contributes gradient into the critic's window.
    for key in ("'q'", "'policy'", "'combiner'", "'mccfr_strat'",
                "'critic'", "'encoder'"):
        assert "_accum_ready({})".format(key) in src, key
    assert "_accum_ready(f'mccfr_adv_{pid}')" in src

    counts = {}
    steps = {k: 0 for k in ("policy", "q", "combiner", "adv", "strat", "value")}

    def ready(k, accum=2):
        n = counts.get(k, 0) + 1
        if n % accum == 0:
            counts[k] = 0
            return True
        counts[k] = n
        return False

    for _ in range(10):
        for head in steps:
            if ready(head):
                steps[head] += 1
    assert all(v == 5 for v in steps.values()), steps


def test_heads_clear_gradients_at_window_start_only():
    """Clearing every call discards half of each accumulation window; never
    clearing leaks gradients between the two optimizers that share
    baseline_network. Window start is the only form that does both."""
    src = _src("neural_cfr_bot/player.py")
    assert "_accum_window_start" in src

    # Only the ACCUMULATING heads are constrained. _train_with_proper_targets
    # zeroes, backwards and steps in one go, so its bare zero_grad() is correct.
    tree = _ast.parse(src)
    accumulating = {"_train_q_network", "_train_policy_network",
                    "_train_strategy_combiner", "_train_deep_mccfr_networks"}
    for node in _ast.walk(tree):
        if not (isinstance(node, _ast.FunctionDef) and node.name in accumulating):
            continue
        body = _ast.get_source_segment(src, node) or ""
        for line in body.splitlines():
            code = line.split("#", 1)[0].strip()
            if code.endswith(".zero_grad()") and "set_to_none" not in code:
                raise AssertionError(
                    "unconditional zero_grad() in accumulating head {}: {}".format(
                        node.name, code))

    for key in ("'q'", "'policy'", "'combiner'", "'mccfr_strat'", "'encoder'"):
        assert "_accum_window_start({})".format(key) in src, key
    assert "_accum_window_start(f'mccfr_adv_{pid}')" in src
    # The critic is excluded: _train_deep_mccfr_networks backprops into the same
    # head out-of-band, so a window-start zero would discard those gradients.
    assert "_accum_window_start('critic')" not in src

    # A window must accumulate exactly accum_steps backwards at full magnitude
    # and never carry a gradient into the next window.
    counts = {}

    def window_start(k):
        return counts.get(k, 0) == 0

    def ready(k, accum=2):
        n = counts.get(k, 0) + 1
        if n % accum == 0:
            counts[k] = 0
            return True
        counts[k] = n
        return False

    grad, steps = 0.0, 0
    for _ in range(8):
        if window_start("q"):
            assert grad == 0.0, "gradient carried across a window boundary"
            grad = 0.0
        grad += 1.0 / 2  # loss is pre-divided by accum_steps
        if ready("q"):
            assert abs(grad - 1.0) < 1e-9, grad
            grad = 0.0
            steps += 1
    assert steps == 4


def test_no_optimizer_shares_parameters_with_another():
    """baseline_network must have exactly one owner.

    It is a submodule of policy_network, so handing policy_network.parameters()
    wholesale to policy_optimizer puts the critic under two optimizers, each
    stepping on the other's partial gradients.
    """
    src = _src("neural_cfr_bot/player.py")
    assert "_critic_ids" in src, "policy_optimizer no longer excludes the critic"
    assert "id(q) not in _critic_ids" in src
    # The wholesale form is the bug; it must not come back.
    for line in src.splitlines():
        code = line.split("#", 1)[0]
        if "policy_optimizer = optim" in code:
            continue
        assert not ("list(self.policy_network.parameters())" in code
                    and "_critic_ids" not in code), line


def test_critic_steps_in_every_stage():
    """Once the critic left policy_optimizer it had to be stepped centrally:
    _train_deep_mccfr_networks never runs in stages 1-2, and the only other
    stepper has no call sites."""
    src = _src("neural_cfr_bot/player.py")
    tree = _ast.parse(src)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef) and node.name == "_train_networks":
            body = _ast.get_source_segment(src, node) or ""
            assert "self.value_optimizer.step()" in body, \
                "critic is not stepped by _train_networks"
            assert "_accum_ready('critic')" in body
            return
    raise AssertionError("_train_networks not found")


def test_encoder_is_stepped_once_after_all_heads():
    """Stepping it inside a head mutates params mid-graph and breaks the next backward."""
    src = _src("neural_cfr_bot/player.py")
    real = [l for l in src.splitlines()
            if "encoder_optimizer.step()" in l and not l.strip().startswith("#")]
    assert len(real) == 1, real
    assert "retain_graph=True" in src


def test_q_bootstrap_uses_the_same_encoder_as_states():
    assert "next_state_data" in _src("neural_cfr_bot/player.py")


def test_deleted_dead_code_stays_deleted():
    src = _src("neural_cfr_bot/player.py")
    for sym in ("CFRNodeData", "cfr_nodes", "_update_cfr_regrets",
                "_update_opponent_hand_range", "_train_opponent_model",
                "HierarchicalLSTMOpponentModel", "opponent_model"):
        assert sym not in src, sym


def test_no_orphaned_decorator_before_attention_block():
    """An AST delete once left @dataclass attached to MultiHeadAttentionBlock."""
    tree = _ast.parse(_src("neural_cfr_bot/player.py"))
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef) and node.name == "MultiHeadAttentionBlock":
            assert node.decorator_list == []
            return
    raise AssertionError("MultiHeadAttentionBlock not found")


# ------------------------------------------------------------------- scripts

def test_slurm_scripts_are_lf_and_parse():
    scripts = sorted(ROOT.glob("*.slurm"))
    assert scripts, "no slurm scripts found"
    have_bash = subprocess.run(["bash", "-c", "exit 0"],
                               capture_output=True).returncode == 0
    for s in scripts:
        assert b"\r\n" not in io.open(s, "rb").read(), "{} has CRLF".format(s.name)
        if have_bash:
            # Pass a bare filename with cwd=ROOT: an MSYS bash on Windows cannot
            # resolve a "C:/..." drive-letter path.
            r = subprocess.run(["bash", "-n", s.name], cwd=str(ROOT),
                               capture_output=True)
            assert r.returncode == 0, "{}: {}".format(s.name, r.stderr.decode()[:200])


def test_deepcfr_slurm_rewrites_player1_unconditionally():
    """The old s.replace('... "mccfr_bot" ...') silently no-opped on any other bot."""
    src = _src("train_deepcfr.slurm")
    assert "PLAYER_1_PATH" in src and "re.subn" in src
    # The defect was a literal replace() keyed on the old path. Comments may
    # mention it; an executable line must not perform it.
    for line in src.splitlines():
        code = line.split("#", 1)[0]
        assert not ("s.replace(" in code and "mccfr_bot" in code), line
