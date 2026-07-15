from benchmarks.workloads import WORKLOADS
from oracle.fixtures import golden_cases


def test_phase_zero_workloads_are_frozen() -> None:
    assert WORKLOADS["W1"].input_tokens == 4096
    assert WORKLOADS["W2"].input_tokens == 32768
    assert all(workload.concurrency == (1, 4) for workload in WORKLOADS.values())


def test_golden_matrix_covers_slot_reuse_and_batch_four() -> None:
    cases = golden_cases()
    assert any(case.phase == "slot_reuse" for case in cases)
    assert any(case.batch_size == 4 for case in cases)
