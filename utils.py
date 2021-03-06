"""
utils.py
Miscellaneous utilities (color picking; progress bars, etc.)
"""

import os
import pickle
import warnings

def ordinal(n):
  """
  Returns the ordinal string for an integer.
  """
  if 11 <= n <= 19:
    return str(n) + "th"
  s = str(n)
  last = int(s[-1])
  if 1 <= last <= 3:
    return s + ("st", "nd", "rd")[last-1]
  return s + "th"

PR_INTRINSIC = 0
PR_CHARS = "▁▂▃▄▅▆▇█▇▆▅▄▃▂"
def prbar(progress, debug=print, interval=1, width=65):
  """
  Prints a progress bar. The argument should be a number between 0 and 1. Put
  this in a loop without any other printing and the bar will fill up on a
  single line. To print stuff afterwards, use an empty print statement after
  the end of the loop to move off of the progress bar line.

  The output will be sent to the given 'debug' function, which is just "print"
  by default.
  
  If an 'interval' value greater than 1 is given, the bar will only be printed
  every interval calls to the function.
  
  'width' may also be specified to determine the width of the bar in characters
  (but note that 6 extra characters are printed, so the actual line width will
  be width + 6).
  """
  global PR_INTRINSIC
  ic = PR_CHARS[PR_INTRINSIC]
  PR_INTRINSIC = (PR_INTRINSIC + 1) % len(PR_CHARS)
  if PR_INTRINSIC % interval != 0:
    return
  sofar = int(width * progress)
  left = width - sofar - 1
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
  """
  Resets the automatic color counter.
  """
  global CURRENT_COLOR
  CURRENT_COLOR = 0

def pick_color(i=None, mute=False, both=False):
  """
  Picks a color, by default via an automatic counter (see reset_color) or via a
  given index. The palette is defined above, and doesn't really have any
  special properties like perceptual uniformity or good grayscale or colorblind
  robustness. If "mute" is given, a desaturated version of the normal color is
  retuned, and if "both" is given, both the normal and desaturated versions are
  returned as a pair.
  """
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
  """
  Returns a debug function, which will be a no-op if "quiet" is True, and print
  otherwise. The function accepts any arguments in either case, so calling it
  won't be an error.
  """
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

def twolevel_default_params(defaults):
  """
  Works like default_params, but for convenience allows grouping parameters
  into sub-dictionaries. If a named parameter is given which matches the name
  of a sub-dictionary and whose value is a dictionary, the default dictionary
  and the given dictionary will be merged, instead of overwriting the default
  sub-dictionary (if the given value isn't a dictionary, it will be used and
  the sub-dictionary will be discarded).
  """
  def wrap(function):
    def withargs(*args, **kwargs):
      merged = {}
      merged.update(defaults)
      for k, v in kwargs.items():
        if type(v) == dict and k in merged and type(merged[k]) == dict:
          merged[k].update(v)
        else:
          merged[k] = v
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
  """
  Runs a function with warnings suppressed. Suppresses only the given warning
  unless "all" is given, in which case it suppresses all warnings.
  """
  if filter_out == "all":
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      return f(*args, **kwargs)
  else:
    with warnings.catch_warnings():
      for wt in filter_out:
        warnings.simplefilter("ignore", wt)
      return f(*args, **kwargs)

def sidak_alpha(desired_confidence, number_of_tests):
  """
  Computes a corrected alpha value for multiple comparisons using the Šidák
  correction.
  """
  # TODO: Bonferroni correction? That requires a whole other algorithm though.
  return 1 - (
    1 - desired_confidence
  )**(1 / number_of_tests)
