import pytest

SEED = 398942280


@pytest.fixture(scope='function', autouse=True)
def fix_seed(request):
    # Ensures test consistency
    import numpy
    numpy.random.seed(SEED)
