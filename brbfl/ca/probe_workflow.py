"""Audit-only local training before ordinary aggregate publication."""

from copy import deepcopy
from time import monotonic

from brbfl.ledger.runtime import get_runtime_ledger
from brbfl.validation.byzantine_gate import parameter_hash, wait_for_admitted_contributors


def recovery_probe_barrier(state, learner, timeout: float, attack=None) -> None:
    """Restore the parent model before releasing any aggregate publisher."""
    runtime = get_runtime_ledger()
    if runtime is None or not runtime.config.recovery_probes_enabled:
        return
    number = int(state.round)
    subjects = runtime.recovery_probe_subjects(number)
    if not subjects:
        return
    try:
        if state.addr in subjects:
            if learner is None:
                raise RuntimeError("probe training requires a learner")
            start = monotonic()
            saved = deepcopy(learner.get_model())
            parent = parameter_hash(saved.get_parameters())
            audit = {"scheduled": True, "trained": False, "submitted": False, "parent_model_hash": parent}
            state.trainer_role_evidence[number]["recovery_probe"] = audit
            try:
                learner.fit()
                audit["trained"] = True
                parameters = [p.copy() for p in learner.get_model().get_parameters()]
                # The audited wrapper's ordinary publication path has aggregate
                # visibility invariants; probes have their own audit record.
                underlying = getattr(attack, "attack", attack)
                active = getattr(attack, "active_in_round", lambda r: True)(number) if attack is not None else False
                if active:
                    if underlying is None or not hasattr(underlying, "manipulate_update"):
                        raise RuntimeError("unsupported probe attack")
                    parameters = underlying.manipulate_update(parameters)
                audit["attack_applied"] = active
                audit["submitted_model_hash"] = parameter_hash(parameters)
            finally:
                learner.set_model(saved)
            audit["restored_parent_hash"] = parameter_hash(learner.get_model().get_parameters())
            if audit["restored_parent_hash"] != parent:
                raise RuntimeError("probe training failed to restore the parent")
            wait_for_admitted_contributors(number, timeout)
            receipt = runtime.submit_recovery_probe(number, state.addr, parent, parameters)
            audit.update(submitted=True, decision_hash=receipt["decision"]["decision_hash"], elapsed_seconds=monotonic() - start)
            runtime.complete_recovery_probe(number, state.addr)
        runtime.wait_for_recovery_probes(number, timeout)
    except Exception as exc:
        runtime.fail_recovery_probes(number, exc)
        raise
