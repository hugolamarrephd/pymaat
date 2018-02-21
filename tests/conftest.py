import pytest

SEED = 12334567

# @pytest.fixture(params=[np.float_,np.int_,np.bool_], scope='function')
# def dtype(request):
#     return request.param

@pytest.fixture(scope='function')
def random(shape, dtype):
    import numpy as np
    if dtype == np.float_:
        return _random_normal(shape)
    elif dtype == np.int_:
        return _random_int(shape)
    elif dtype == np.bool_:
        return _random_bool(shape)
    else:
        return np.array(_random_normal(shape), dtype=dtype)

@pytest.fixture(scope='function')
def random_int(shape):
    """
    Random integer variates (between 0 and prod(shape))
        of specified shape.
    Rem. returns same numbers for each run (for a given shape)
        to ensure test consistency.
    """
    return _random_int(shape)

def _random_int(shape):
    import numpy as np
    np.random.seed(SEED) # Ensures test consistency
    return np.random.randint(low=0, high=np.prod(shape), size=shape)

@pytest.fixture(scope='function')
def random_normal(shape):
    """
    Random variates (normal(0,1)) of specified shape.

    Rem. returns same numbers for each run (for a given shape)
        to ensure test consistency.
    """
    return _random_normal(shape)

def _random_normal(shape):
    import numpy as np
    np.random.seed(SEED) # Ensures test consistency
    return np.random.normal(size=shape)

@pytest.fixture(scope='function')
def random_bool(shape):
    """
    Boolean random variates of specified shape.

    Rem. returns same values for each run (for a given shape)
        to ensure test consistency.
    """
    return _random_bool(shape)

def _random_bool(shape):
    import numpy as np
    np.random.seed(SEED) # Ensures test consistency
    return np.random.choice([True, False], size=shape)

@pytest.fixture(scope='function')
def random_gamma(shape):
    """
    Positive random variates (gamma(1,1)) of specified shape.

    Rem. returns same numbers for each run (for a given shape)
        to ensure test consistency.
    """
    import numpy as np
    np.random.seed(SEED) # Ensures test consistency
    return np.random.gamma(1., size=shape)

@pytest.fixture(scope = 'function')
def zeros(shape, dtype):
    import numpy as np
    return np.full(shape, 0, dtype=dtype)

@pytest.fixture(scope = 'function')
def ones(shape, dtype):
    import numpy as np
    return np.full(shape, 1, dtype=dtype)

@pytest.fixture(scope = 'function')
def inf(shape):
    import numpy as np
    return np.full(shape, np.inf)

@pytest.fixture(scope = 'function')
def minf(shape):
    import numpy as np
    return np.full(shape, -np.inf)

@pytest.fixture(scope = 'function')
def nan(shape):
    import numpy as np
    return np.full(shape, np.nan)

@pytest.fixture(scope = 'function')
def true(shape):
    import numpy as np
    return np.full(shape, True)

@pytest.fixture(scope = 'function')
def false(shape):
    import numpy as np
    return np.full(shape, False)
