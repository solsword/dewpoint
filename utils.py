"""
utils.py
Miscellaneous utilities (color picking; progress bars, etc.)
"""

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
  (0.0, 0.8, 0.1),
  (0.7, 0.3, 0.8),
  (0.0, 0.7, 0.7),
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
  def wrap(function):
    def withargs(*args, **kwargs):
      merged = {}
      merged.update(defaults)
      merged.update(kwargs)
      return function(*args, **merged)
    return withargs
  return wrap
