import time

def timethis(number=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
           start = time.perf_counter()
           t = np.empty(number)
           for i in range(number):
               r = func(*args, **kwargs)
               t[i] = time.perf_counter() - start
           print('{}.{} : {} sec'.format(
               func.__module__,
               func.__name__,
               t.mean()))
           return r
        return wrapper
    return decorator

@contextmanager
def timeblock(label):
      start = time.perf_counter()
      try:
          yield
      finally:
          end = time.perf_counter()
          print('{} : {} sec'.format(label, end-start))
