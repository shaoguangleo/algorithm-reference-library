"""
See comment by Matt Alcock at http://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
"""

from functools import wraps
from time import time
import cProfile, pstats, io

def timing(f, verbose=False):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        if verbose:
            print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        else:
            print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result
    return wrap

class Profiling(object):
    """ Some quick profiling, as a block object """
    def __init__(self, name='', keys=['cumulative'], restrictions=[10]):
        self.name = name
        self.keys = keys
        self.restrictions = restrictions
    def __enter__(self):
        self.pr = cProfile.Profile()
        self.pr.enable()
    def __exit__(self, *args, **kw):
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(self.pr, stream=s).sort_stats(*self.keys)
        ps.print_stats(*self.restrictions)
        print('== Profile for %s' % self.name)
        print(s.getvalue())

def profiling(f, *pargs, **pkw):
    """ Some quick profiling, as a function wrapper """
    @wraps(f)
    def wrap(*args, **kw):
        with Profiling("func:%r args:[%r, %r]" % (f.__name__, args, kw),
                       *pargs, **pkw):
            return f(*args, **kw)
    return wrap
