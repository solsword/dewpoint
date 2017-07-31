"""
utils.py
Miscellaneous utilities (color picking; progress bars, etc.)
"""

import os
import pickle
import warnings

PR_INTRINSIC = 0
PR_CHARS = "▁▂▃▄▅▆▇█▇▆▅▄▃▂"
def prbar(progress, debug=print):
  global PR_INTRINSIC
  pbwidth = 65
  sofar = int(pbwidth * progress)
  left = pbwidth - sofar - 1
  ic = PR_CHARS[PR_INTRINSIC]
  PR_INTRINSIC = (PR_INTRINSIC + 1) % len(PR_CHARS)
  debug("\r[" + "="*sofar + ">" + "-"*left + "] (" + ic + ")", end="")

POINT_COLOR = (0, 0, 0)
PLOT_COLORS = [
  (0.0, 0.2, 0.8),
  (0.8, 0.0, 0.0),
  (1.0, 0.7, 0.0),
  (0.2, 0.9, 0.0),
  (0.7, 0.3, 0.8),
  (0.0, 0.7, 0.7),
  (0.5, 0.4, 0.0),
  (0.0, 0.0, 0.0),
  (0.5, 0.5, 0.5),
]
DESATURATED_COLORS = [
  [(ch*2 + (sum(c) / len(c)))/3.5 for ch in c]
    for c in PLOT_COLORS
]
CURRENT_COLOR = 0

def reset_color():
  global CURRENT_COLOR
  CURRENT_COLOR = 0

def pick_color(i=None, mute=False, both=False):
  global CURRENT_COLOR
  if i == None:
    i = CURRENT_COLOR
    CURRENT_COLOR += 1
  if both:
    return (
      PLOT_COLORS[i % len(PLOT_COLORS)],
      DESATURATED_COLORS[i % len(DESATURATED_COLORS)]
    )
  elif mute:
    return DESATURATED_COLORS[i % len(DESATURATED_COLORS)]
  else:
    return PLOT_COLORS[i % len(PLOT_COLORS)]

def get_debug(quiet):
  if quiet:
    def debug(*args, **kwargs):
      pass
  else:
    def debug(*args, **kwargs):
      print(*args, **kwargs)
  return debug

def default_params(defaults):
  """
  Returns a decorator which attaches the given dictionary as default parameters
  for the decorated function. Any keyword arguments supplied manually will
  override the provided defaults, and non-keyword arguments are passed through
  normally.
  """
  def wrap(function):
    def withargs(*args, **kwargs):
      merged = {}
      merged.update(defaults)
      merged.update(kwargs)
      return function(*args, **merged)
    return withargs
  return wrap

def multilevel_default_params(defaults):
  """
  Works like default_params, but for convenience allows grouping parameters
  into sub-dictionaries. If a named parameter is given which matches the name
  of an item in exactly one sub-dictionary, that item will be updated (and the
  parameter is retained in the top-level dictionary as well). If an item
  matches in multiple sub-dictionaries, no action is taken.
  """
  def wrap(function):
    def withargs(*args, **kwargs):
      merged = {}
      merged.update(defaults)
      sds = []
      for v in merged.values():
        if type(v) == dict:
          sds.append(v)
      for ak in kwargs:
        hits = [ak in sd for sd in sds]
        if len(hits) == 1:
          hits[0][ak] = kwargs[ak]
      merged.update(kwargs)
      return function(*args, **merged)

    return withargs

  return wrap

CACHE_DIR = ".cache" # directory for cache files
DISABLE_CACHING = False

def cache_name(name, typ="pkl"):
  """
  Computes a filename for a cache value of the given type.
  """
  return os.path.join(CACHE_DIR, name + '.' + typ)

def is_cached(name, typ="pkl"):
  """
  Checks whether anything is cached for the given name/type.
  """
  return os.path.exists(cache_name(name, typ))

def load_cache(name, typ="pkl"):
  """
  Loads a value from the cache for the given name/type.
  """
  filename = cache_name(name, typ)
  if typ == "str":
    with open(filename, 'r') as fin:
      return fin.read()
  elif typ == "pkl":
    with open(filename, 'rb') as fin:
      return pickle.load(fin)
  elif typ == "h5":
    import keras
    return keras.models.load_model(filename)
  else:
    raise ValueError("Invalid type '{}'.".format(typ))

def store_cache(value, name, typ="pkl"):
  """
  Stores the given value in the cache under the given name/type.
  """
  if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR, mode=0o775)
  filename = cache_name(name, typ)
  if typ == "str":
    with open(filename, 'w') as fout:
      fout.write(value)
  elif typ == "pkl":
    with open(filename, 'wb') as fout:
      pickle.dump(value, fout)
  elif typ == "h5":
    value.save(filename)
  else:
    raise ValueError("Invalid type '{}'.".format(typ))

def cached_value(compute, name, typ="pkl", override=False, debug=None):
  """
  Takes a name, a type, and a compute function. If there's a cached value under
  the given name/type, just loads and returns it, otherwise (or if the given
  "override" value is True), calls the compute function to generate a new value
  and caches that value before returning it. It calls the given debug function
  with messages about what it's doing.
  """
  debug = debug or (lambda msg: None)
  if not DISABLE_CACHING and not override and is_cached(name, typ):
    debug("Loading cached '{}'...".format(name))

    try:
      result = load_cache(name, typ)
      debug("...done.")
      return result
    except:
      debug("...failed to load cached '{}'!")

  debug("Computing new '{}'...".format(name))
  val = compute()
  debug("...done computing '{}'. Caching value.".format(name))
  store_cache(val, name, typ)
  return val

def cached_values(compute, names, types, override=False, debug=None):
  """
  Works exactly like cached_value, but the compute function should return a
  tuple of results, which are cached separately under names/types from the
  given tuples. Unless every value can be loaded from the cache, all values
  will be recomputed.
  """
  debug = debug or (lambda msg: None)
  z = list(zip(names, types))
  allnames = "', '".join(names)

  if (
    not DISABLE_CACHING
and not override
and all(is_cached(n, t) for (n, t) in z)
  ):
    debug("Loading cached '{}'...".format(allnames))
    try:
      results = [load_cache(n, t) for (n, t) in z ]
      debug("...done.")
      return results
    except:
      debug("...failed to load cached '{}'!".format(allnames))

  else:
    debug("Computing new '{}'...".format(allnames))

    values = compute()

    debug("...done computing '{}'. Caching values.".format(allnames))

    for i, (n, t) in enumerate(z):
      store_cache(values[i], n, t)

    return values

def toggle_caching(on=None):
  """
  Globally toggles caching to the opposite of the current value, or to the
  specified on value if given as True or False.
  """
  global DISABLE_CACHING
  if on is None:
    DISABLE_CACHING = not DISABLE_CACHING
  else:
    DISABLE_CACHING = bool(on)

def run_strict(f, *args, **kwargs):
  """
  Runs a function with warnings as errors.
  """
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    return f(*args, **kwargs)

def run_lax(filter_out, f, *args, **kwargs):
  if filter_out == "all":
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      return f(*args, **kwargs)
  else:
    with warnings.catch_warnings():
      for wt in filter_out:
        warnings.simplefilter("ignore", wt)
      return f(*args, **kwargs)
