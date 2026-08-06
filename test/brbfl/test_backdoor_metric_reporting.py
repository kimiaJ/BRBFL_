"""Tests for honest backdoor metric naming and suppression."""

from brbfl.evaluation.metrics import backdoor_evaluation_configured


def test_asr_is_not_configured_for_clean_or_sign_flipping_attacks():
    """A target-class rate without trigger semantics must not be called ASR."""

    class SignFlipping:
        scale = -3.0

    assert backdoor_evaluation_configured(None) is False
    assert backdoor_evaluation_configured(SignFlipping()) is False


def test_asr_requires_explicit_trigger_evaluation_semantics():
    """Complete trigger metadata enables genuine ASR evaluation."""

    class Backdoor:
        trigger_size = 3
        trigger_value = 1.0
        target_class = 2

        def poison_batch(self, batch):
            return batch

    assert backdoor_evaluation_configured(Backdoor()) is True
