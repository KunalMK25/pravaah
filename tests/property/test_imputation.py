"""
Property 2: Missing Value Imputation Completeness

Validates: Requirements 1.2
"""
import math
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from flood_risk_zonation.utils.validation import impute_missing_values

_values_strategy = st.lists(
    st.one_of(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        st.just(float("nan")),
    ),
    min_size=4,
    max_size=50,
)


@given(values=_values_strategy)
@settings(max_examples=20)
def test_imputation_completeness_1d(values):
    """Property 2: After imputation, no NaN values remain in a 1D array."""
    arr = np.array(values, dtype=np.float64)
    result = impute_missing_values(arr)
    assert not np.any(np.isnan(result))


@given(values=_values_strategy)
@settings(max_examples=20)
def test_imputation_completeness_2d(values):
    """Property 2: After imputation, no NaN values remain in a 2D array."""
    n = len(values)
    ncols = max(2, int(math.sqrt(n)))
    nrows = max(2, math.ceil(n / ncols))
    padded = values + [float("nan")] * (nrows * ncols - n)
    arr = np.array(padded, dtype=np.float64).reshape(nrows, ncols)
    result = impute_missing_values(arr)
    assert not np.any(np.isnan(result))


@given(values=st.lists(st.just(float("nan")), min_size=4, max_size=20))
@settings(max_examples=20)
def test_imputation_all_nan_fills_with_zero(values):
    """When all values are NaN, imputation should fill with 0.0."""
    arr = np.array(values, dtype=np.float64)
    result = impute_missing_values(arr)
    assert not np.any(np.isnan(result))
    assert np.all(result == 0.0)
