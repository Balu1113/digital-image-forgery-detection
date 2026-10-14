import pytest

from bandit import ContextualBandit


def _bandit(path=None, num_states=10, actions=(0.35, 0.45, 0.55, 0.65)):
    return ContextualBandit(num_states=num_states, actions=actions, path=path)


def test_state_for_buckets():
    bandit = _bandit()
    assert bandit.state_for(0.0) == 0
    assert bandit.state_for(0.49) == 4
    assert bandit.state_for(0.5) == 5
    assert bandit.state_for(0.999999) == 9
    assert bandit.state_for(1.0) == 9  # clamped into the last bucket
    assert bandit.state_for(-3) == 0


def test_select_explores_untried_actions_first():
    bandit = _bandit()
    state = 3
    assert bandit.select(state) == 0
    bandit.observe(state, 0, 1.0)
    assert bandit.select(state) == 1
    bandit.observe(state, 1, -1.0)
    assert bandit.select(state) == 2
    bandit.observe(state, 2, 1.0)
    assert bandit.select(state) == 3
    bandit.observe(state, 3, 1.0)
    # Everything tried: highest mean wins (actions 0 and 2 tie at +1 -> index 0).
    assert bandit.select(state) == 0
    # Untouched states are independent.
    assert bandit.select(state + 1) == 0


def test_observe_updates_means_and_count_flag():
    bandit = _bandit(num_states=2, actions=(0.4, 0.6))
    bandit.observe(0, 0, 1.0)
    bandit.observe(0, 0, -1.0)
    assert bandit.mean(0, 0) == pytest.approx(0.0)

    # A correction adjusts the reward without counting as a fresh pull.
    bandit.observe(0, 0, -1.0, count=False)
    assert bandit.mean(0, 0) == pytest.approx(-0.5)
    assert bandit.summary()["states"][0]["counts"][0] == 2

    assert bandit.mean(1, 1) is None


def test_observe_validates_input():
    bandit = _bandit(num_states=2, actions=(0.4, 0.6))
    with pytest.raises(ValueError):
        bandit.observe(9, 0, 1.0)
    with pytest.raises(ValueError):
        bandit.observe(0, 5, 1.0)
    with pytest.raises(ValueError):
        bandit.observe(0, 0, 5.0)
    with pytest.raises(ValueError):
        bandit.select(42)


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "policy.json"
    bandit = _bandit(path=path, num_states=2, actions=(0.4, 0.6))
    bandit.observe(1, 1, 1.0)
    bandit.observe(1, 1, 1.0)
    assert path.exists()

    restored = _bandit(path=path, num_states=2, actions=(0.4, 0.6))
    assert restored.mean(1, 1) == pytest.approx(1.0)
    assert restored.select(0) == 0


def test_corrupt_policy_file_is_ignored(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text("not json {", encoding="utf-8")
    bandit = _bandit(path=path, num_states=2, actions=(0.4, 0.6))
    assert bandit.mean(0, 0) is None


def test_summary_shape():
    bandit = _bandit(num_states=3, actions=(0.4, 0.6))
    bandit.observe(0, 1, 1.0)
    summary = bandit.summary()
    assert summary["actions"] == [0.4, 0.6]
    assert summary["pulls"] == 1
    assert summary["states"][0]["best_action"] == 1
    assert summary["states"][1]["best_action"] is None
    assert summary["states"][0]["means"][1] == pytest.approx(1.0)
