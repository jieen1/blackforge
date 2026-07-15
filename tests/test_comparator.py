from oracle.comparator import compare_values


def test_compare_values_reports_identical_outputs() -> None:
    result = compare_values([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], top_k=2)

    assert result.max_abs_error == 0.0
    assert result.cosine_similarity == 1.0
    assert result.top_k_agreement == 1.0
    assert result.passes(max_abs_error=0.0, min_cosine=1.0, min_top_k=1.0)


def test_compare_values_rejects_different_shapes() -> None:
    try:
        compare_values([1.0], [1.0, 2.0])
    except ValueError as error:
        assert "sizes differ" in str(error)
    else:
        raise AssertionError("expected a shape mismatch")
