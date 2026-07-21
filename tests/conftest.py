"""Pytest configuration.

test_real_world.py, test_api_compat.py, and test_e2e_256k_longctx.py are
integration/E2E test scripts that require a running server. They are excluded
from pytest collection and should be run manually:

    python tests/test_real_world.py [base_url]
    python tests/test_api_compat.py --base-url http://127.0.0.1:8000
    python tests/test_e2e_256k_longctx.py --base-url http://127.0.0.1:8000
"""

collect_ignore = [
    "test_real_world.py",
    "test_api_compat.py",
    "test_e2e_256k_longctx.py",
]
