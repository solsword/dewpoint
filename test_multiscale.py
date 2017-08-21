#!/usr/bin/env python3
"""
Tests for multiscale.py.
"""

import utils

import sys
import pickle
import random
import math

import unionfind as uf

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import lines as ml
from matplotlib import collections as mc
from matplotlib import animation as ma

from scipy.optimize import curve_fit
from scipy.spatial.distance import euclidean

import multiscale

def gcluster(loc, size, n):
  return np.stack(
    [
      np.random.normal(l, s, size=n)
        for (l, s) in zip(loc, size)
    ],
    axis=-1
  )

def gring(loc, size, width, n):
  functions = []
  for i in range(len(size)):
    functions.append([np.cos, np.sin][i%2])

  return np.stack(
    [
      l + (
        s + np.random.normal(0, width, n)
      ) * func(np.linspace(0, 2*math.pi, n))
        for (l, s, func) in zip(loc, size, functions)
    ],
    axis=-1
  )

def gline(fr, to, var, n):
  return np.concatenate(
    [
      np.linspace(fr[0], to[0], n).reshape((n, 1)),
      np.linspace(fr[1], to[1], n).reshape((n, 1))
    ],
    axis=1
  ) + gcluster((0.0, 0.0), (var, var), n)

test_cases = {
  "scatter": np.random.uniform(0, 1, size=(100, 2)),
  "normal": gcluster((0.5, 0.5), (0.3, 0.3), 100),
  "larger_normal": gcluster((0.5, 0.5), (1.0, 1.0), 100),
  "ring": gring((0.5, 0.5), (0.4, 0.6), 0.03, 100),
  "pair": np.concatenate(
    [
      gcluster((0.2, 0.6), (0.05, 0.05), 100),
      gcluster((0.7, 0.3), (0.07, 0.07), 100),
    ]
  ),
  "ring_pair": np.concatenate(
    [
      gring((0.2, 0.6), (0.2, 0.15), 0.05, 100),
      gring((0.7, 0.3), (0.1, 0.3), 0.05, 100),
    ]
  ),
  "triple": np.concatenate(
    [
      gcluster((0.2, 0.2), (0.03, 0.03), 100),
      gcluster((0.1, 0.3), (0.03, 0.03), 100),
      gcluster((0.9, 0.4), (0.08, 0.08), 100),
    ]
  ),
  "triple_far": np.concatenate(
    [
      gcluster((0.2, 0.2), (0.03, 0.03), 100),
      gcluster((0.1, 0.8), (0.03, 0.03), 100),
      gcluster((0.9, 0.4), (0.08, 0.08), 100),
    ]
  ),
  "quadruple": np.concatenate(
    [
      gcluster((0.2, 0.2), (0.03, 0.03), 100),
      gcluster((0.1, 0.8), (0.03, 0.03), 100),
      gcluster((0.6, 0.4), (0.08, 0.12), 100),
      gcluster((0.9, 0.9), (0.15, 0.15), 100),
    ]
  ),
  "concentric": np.concatenate(
    [
      gring((0.5, 0.5), (0.6, 0.6), 0.04, 100),
      gring((0.5, 0.5), (0.2, 0.2), 0.05, 100),
    ]
  ),
  "quad_encircled": np.concatenate(
    [
      gcluster((0.2, 0.2), (0.06, 0.06), 80),
      gcluster((0.8, 0.8), (0.04, 0.04), 80),
      gcluster((0.2, 0.8), (0.06, 0.04), 80),
      gcluster((0.8, 0.2), (0.04, 0.06), 80),
      gcluster((0.5, 0.5), (0.6, 0.6), 250),
      gring((0.45, 0.55), (0.9, 0.9), 0.07, 200),
    ]
  ),
  "overlapping": np.concatenate(
    [
      gcluster((0.5, 0.5), (0.06, 0.06), 80),
      gcluster((0.8, 0.8), (0.36, 0.25), 160),
      gcluster((0.4, 0.4), (0.60, 0.60), 240),
    ],
    axis=0
  ),
  "pair_with_line": np.concatenate(
    [
      gcluster((0.2, 0.6), (0.05, 0.05), 100),
      gcluster((0.7, 0.3), (0.07, 0.07), 100),
      gline((0, 0), (1, 1), 0.015, 50)
    ],
    axis=0
  ),
  "complex": np.concatenate(
    [
      gcluster((0.5, 0.5), (0.5, 0.5), 160),
      gcluster((0.7, 0.3), (0.3, 0.5), 120),
      gcluster((0.3, 0.4), (0.1, 0.08), 100),
      gcluster((1.3, 1.5), (0.4, 0.1), 120)
    ],
    axis=0
  ),
  "A": np.concatenate(
    [
      gline((0.1, 0.1), (0.5, 0.9), 0.02, 100),
      gline((0.9, 0.1), (0.5, 0.9), 0.02, 100),
      gline((0.3, 0.5), (0.7, 0.5), 0.02, 100),
    ],
    axis=0
  ),
  "Ab": np.concatenate(
    [
      gline((0.1, 0.1), (0.5, 0.9), 0.02, 100),
      gline((0.9, 0.1), (0.5, 0.9), 0.02, 100),
      gline((0.3, 0.5), (0.7, 0.5), 0.02, 100),
      gline((1.1, 0.1), (1.1, 0.9), 0.02, 100),
      gring((1.3, 0.3), (0.2, 0.2), 0.01, 100),
    ],
    axis=0
  ),
  "X": np.concatenate(
    [
      np.concatenate(
        [
          np.arange(0, 1, 0.02).reshape(50, 1),
          np.arange(0, 1, 0.02).reshape(50, 1)
        ],
        axis=1
      ) + gcluster((0.0, 0.0), (0.012, 0.012), 50),
      np.concatenate(
        [
          np.arange(0, 1, 0.02).reshape(50, 1),
          np.arange(1, 0, -0.02).reshape(50, 1)
        ],
        axis=1
      ) + gcluster((0.0, 0.0), (0.012, 0.012), 50)
    ],
    axis=0
  ),
  "parallel": np.concatenate(
    [
      np.concatenate(
        [
          np.arange(0, 1, 0.02).reshape(50, 1),
          np.arange(0, 1, 0.02).reshape(50, 1)
        ],
        axis=1
      ) + gcluster((0.0, 0.0), (0.012, 0.012), 50),
      np.concatenate(
        [
          np.arange(0, 1, 0.02).reshape(50, 1),
          np.arange(0, 1, 0.02).reshape(50, 1) + 0.4
        ],
        axis=1
      ) + gcluster((0.0, 0.0), (0.012, 0.012), 50)
    ],
    axis=0
  ),
  "triple_helix": np.concatenate(
    (
      np.stack(
        (
          np.cos(np.linspace(0, 2 * math.pi, 100)),
          np.linspace(0, 2 * math.pi, 100),
          np.sin(np.linspace(0, 2 * math.pi, 100)),
        ),
        axis=-1
      ) + gcluster((0, 0, 0), (0.18, 0.09, 0.18), 100),
      np.stack(
        (
          np.cos(2 * math.pi / 3 + np.linspace(0, 2 * math.pi, 100)),
          np.linspace(0, 2 * math.pi, 100),
          np.sin(2 * math.pi / 3 + np.linspace(0, 2 * math.pi, 100)),
        ),
        axis=-1
      ) + gcluster((0, 0, 0), (0.18, 0.09, 0.18), 100),
      np.stack(
        (
          np.cos(4 * math.pi / 3 + np.linspace(0, 2 * math.pi, 100)),
          np.linspace(0, 2 * math.pi, 100),
          np.sin(4 * math.pi / 3 + np.linspace(0, 2 * math.pi, 100)),
        ),
        axis=-1
      ) + gcluster((0, 0, 0), (0.18, 0.09, 0.18), 100),
    ),
    axis=0
  )
}

KERNEL = np.asarray([0.75, 0.75, 1, 1, 1, 1, 1, 0.75, 0.75])
KERNEL /= sum(KERNEL)

PREV_KERNEL = np.asarray([0.0]*5 + [1.0]*6)
PREV_KERNEL /= sum(PREV_KERNEL)

def newfig():
  utils.reset_color()
  plt.figure()

def plot_data(
  data,
  mute=False,
  label=None,
  smooth=False,
  baseline=False,
  stats=False,
):
  c, d = utils.pick_color(both=True)
  if mute:
    c = d
    d = (0.4, 0.4, 0.4)

  plt.axhline(0, color=(0, 0, 0))
  plt.plot(
    data,
    color=c,
    label=label
  )
  if baseline:
    base = np.median(data[:5,:])*baseline
    plt.axhline(base, color=d)

  if smooth:
    thresh = np.convolve(
      data,
      PREV_KERNEL,
      mode="same"
    ) * smooth
    plt.plot(thresh, color=d)

  if stats:
    mean = np.mean(data)
    std = np.std(data)
    plt.axhline(mean, color=d)
    plt.axhline(mean + std, color=d)
    plt.axhline(mean - std, color=d)

def plot_property(points, properties, ax=None, title=None):
  print("Plotting {} points..".format(len(points)))
  projected = points[:,:2]

  if ax == None:
    fig, ax = plt.subplots()
    if title:
      plt.title(title)
    else:
      plt.title("Points")
    plt.axis("equal")
  elif title:
    ax.set_title(title)

  cmap = plt.get_cmap("plasma")
  colors = np.apply_along_axis(cmap, 0, properties)

  ax.scatter(projected[:,0], projected[:,1], s=1.0, c=colors)

  print("...done plotting points.")

def plot_clusters(points, clusters, ax=None, title=None, show_quality=None):
  print("Plotting {} clusters..".format(len(clusters)))

  # Plot in 2 dimensions; ignore any other dimensions of the data
  projected = points[:,:2]

  if ax == None:
    fig, ax = plt.subplots()
    if title:
      plt.title(title)
    else:
      plt.title("Clustering Results")
    plt.axis("equal")
  elif title:
    ax.set_title(title)


  ax.scatter(projected[:,0], projected[:,1], s=0.8, c=utils.POINT_COLOR)

  by_size = sorted(list(clusters.items()), key=lambda kv: -kv[1]["size"])

  cmap = {}
  for ck, cl in by_size:
    if ck not in cmap:
      clcolor = utils.pick_color()
      cmap[ck] = clcolor
    else:
      clcolor = cmap[ck]

    edges = []
    colors = []
    widths = []
    styles = []

    if not cl["edges"]:
      continue

    for (fr, to, d, q) in cl["edges"]:
      edges.append((projected[fr], projected[to]))
      colors.append(clcolor)
      if show_quality == "edges":
        widths.append(2*q)
      elif show_quality in cl:
        widths.append(2*cl[show_quality])
      else:
        widths.append(0.4)
      if "outliers" in cl:
        if fr in cl["outliers"] or to in cl["outliers"]:
          styles.append("dotted")
        else:
          styles.append("solid")

    lc = mc.LineCollection(
      edges,
      colors=colors,
      linewidths=widths,
      linestyles=styles
    )
    ax.add_collection(lc)
  print("...done plotting results.")


def plot_reps(points, reps, ax=None, title=None):
  print("Plotting {} groups..".format(len(reps)))

  # Plot in 2 dimensions; ignore any other dimensions of the data
  projected = points[:,:2]

  if ax == None:
    fig, ax = plt.subplots()
    if title:
      plt.title(title)
    else:
      plt.title("Representatives")
    plt.axis("equal")
  elif title:
    ax.set_title(title)


  ax.scatter(projected[:,0], projected[:,1], s=0.8, c=utils.POINT_COLOR)

  by_size = sorted(list(reps.items()), key=lambda kv: -len(kv[1]))

  cmap = {}
  for rep, group in by_size:
    if rep not in cmap:
      clcolor = utils.pick_color()
      cmap[rep] = clcolor
    else:
      clcolor = cmap[rep]

    edges = []
    colors = []
    widths = []

    for p in group:
      edges.append((projected[rep], projected[p]))
      colors.append(clcolor)
      widths.append(0.4)

    lc = mc.LineCollection(
      edges,
      colors=colors,
      linewidths=widths
    )
    ax.add_collection(lc)
  print("...done plotting results.")


def plot_stats(
  name,
  clusters,
  stats,
  sort_by="size",
  normalize=None,
  show_mean=None
):

  collected = {k: [] for k in stats}

  n = len(clusters)

  ordered = sorted(list(clusters.values()), key=lambda cl: cl[sort_by])

  for cl in ordered:
    for st in stats:
      if st in cl:
        collected[st].append(cl[st])
      else:
        collected[st].append(-0.1)

  for st in stats:
    collected[st] = np.asarray(collected[st])
    if normalize and st in normalize:
      upper = np.max(collected[st])
      collected[st] = collected[st] / upper

  plt.figure()
  utils.reset_color()
  for st in stats:
    if show_mean and st in show_mean:
      cp, cs = utils.pick_color(both=True)
      plt.scatter(
        range(n),
        collected[st],
        label=st,
        c=cp,
        s=2.4
      )
      plt.axhline(
        np.mean(collected[st]),
        label="mean_" + st,
        c=cs,
        lw=0.9
      )
    else:
      plt.scatter(
        range(n),
        collected[st],
        label=st,
        c=utils.pick_color(),
        s=2.4
      )

  plt.legend()
  plt.title("{} stats".format(name))

def cluster_sequence(clusters):
  root = multiscale.find_largest(clusters)

  seq = [ { root["id"]: root } ]

  while any(cl["children"] for cl in seq[-1].values()):
    seq.append({})
    for cl in seq[-2].values():
      for child in cl["children"]:
        seq[-1][child["id"]] = child

  return seq

def plot_cluster_sequence(points, clusters):
  seq = cluster_sequence(clusters)
  sqw = int(math.ceil(len(seq)**0.5))
  sqh = sqw
  while len(seq) <= sqw * sqh - sqw:
    sqh -= 1
  if sqw * sqh == 1:
    plot_clusters("0", points, clusters)
  else:
    fig, axes = plt.subplots(sqh, sqw, sharex=True, sharey=True)
    plt.axis("equal")
    for i, cls in enumerate(seq):
      sqx, sqy = i % sqw, i // sqw
      if sqh == 1:
        plot_clusters(str(i), points, cls, ax=axes[sqx])
      else:
        plot_clusters(str(i), points, cls, ax=axes[sqy][sqx])

def animate_mst(points, edges):
  fig, axes = plt.subplots(1,2)
  scax, pltax = axes
  scax.scatter(points[:,0], points[:,1], s=0.5, c="k")
  by_size = sorted(edges, key=lambda e: (e[2], e[0], e[1]))
  segments = [
    (points[fr], points[to])
      for (fr, to, d) in by_size
  ]
  cmap = plt.get_cmap("plasma")

  i = 0
  total = 0

  unions = None
  clusters = None
  colors = None
  lc = None
  pc = None
  domain = []
  scale_ratio = []
  scale_concentration = []

  def init():
    nonlocal unions, clusters, colors, lc, pc, i, total, scale_ratio, scale_concentration
    # The scatter subfigure:
    i = 0
    total = 0
    unions = uf.unionfind(len(points))
    clusters = {
      i: {
        "size": 1,
        "mean": 0,
        "scale": 0,
        "points": { i },
        "edges": set(),
      }
        for i in range(len(points))
    }
    colors = [(1, 1, 1, 0)] * len(by_size)
    lc = mc.LineCollection(
      segments,
      linewidths=0.8,
      color=colors,
      animated=True
    )
    scax.add_collection(lc)

    # The data subfigure:
    domain = []
    scale_ratio = []
    scale_concentration = []
    pc = ml.Line2D(
      [],
      [],
      marker='o',
      ms=1.2,
      c=utils.pick_color(),
      ls='None',
      animated=True
    )
    pltax.add_line(pc)
    pltax.set_xlim(-1, len(points))
    #pltax.set_ylim(0.9, 1.5)
    pltax.set_ylim(0.0, 20)
    return [lc, pc]

  edges_so_far = set()
  def update(frame):
    nonlocal unions, i, total, colors, lc, pc, domain, scale_ratio, scale_concentration, edges_so_far
    if i >= len(by_size):
      return [lc, pc]
    fr, to, d = by_size[i]
    cf = unions.find(fr)
    ct = unions.find(to)
    while cf == ct:
      i += 1
      if i >= len(by_size):
        # ran out of edges
        return [lc, pc]
      fr, to, d = by_size[i]
      cf = unions.find(fr)
      ct = unions.find(to)

    # otherwise add it:
    clf, clt = clusters[cf], clusters[ct]
    i += 1
    total += 1
    domain.append(frame)
    longer = max(clf["scale"], clt["scale"])
    smaller = min(clf["size"], clt["size"])
    if longer == 0 or smaller < 6:
      scr = 1
    else:
      scr = d / longer

    nscr = min(1, (scr - 1)/0.25)

    insig = smaller < 6 and scr < 1.1
    insig = scr < 1.1

    scale_ratio.append(scr)

    nmean = (
      (clf["size"] - 1) * clf["mean"]
    + (clt["size"] - 1) * clt["mean"]
    + d
    ) / (
      (clf["size"] - 1)
    + (clt["size"] - 1)
    + 1
    )
    nsize = clf["size"] + clt["size"]
    npoints = clf["points"] | clt["points"]
    nedges = clf["edges"] | clt["edges"] | { (fr, to, d) }
    del clusters[cf]
    del clusters[ct]

    unions.unite(cf, ct)

    cj = unions.find(fr)
    clusters[cj] = {
      "size": nsize,
      "mean": nmean,
      "scale": d,
      "points": npoints,
      "edges": nedges
    }

    #conc = np.std([ed for (ef, et, ed) in nedges])
    edges_so_far.add( (fr, to, d) )
    conc = 0
    refd = nmean
    for (of, ot, od) in edges_so_far:
      dst = abs(od - refd)
      ndst = (dst / (0.1 * refd))
      if ndst < 1:
        conc += 1 - ndst

    scale_concentration.append(conc)

    #colors[i] = cmap(nscr)
    if insig:
      colors[i] = (0, 0, 0, 0.2)
    else:
      colors[i] = (0, 0.5, 1.0, 1)

    if not insig:
      for e in clf["edges"] | clt["edges"]:
        eidx = by_size.index(e)
        colors[eidx] = (1.0, 0, 0, 0.6)

    lc.set_color(colors)
    #pc.set_data(domain, scale_ratio)
    pc.set_data(domain, scale_concentration)

    print("frame", frame, "/", len(points))
    return [lc, pc]

  animation = ma.FuncAnimation(
    fig,
    update,
    interval=1,
    frames=np.arange(0, len(points) - 1),
    repeat=False,
    init_func=init,
    blit=True
  )

  fig.set_size_inches(17, 8, forward=True)

  plt.show()

  #plt.clf()
  #plt.hist([d for (fr, to, d) in edges_so_far], 20)

  #plt.show()

IRIS_DATA = np.asarray([
  [5.1,3.5,1.4,0.2],
  [4.9,3.0,1.4,0.2],
  [4.7,3.2,1.3,0.2],
  [4.6,3.1,1.5,0.2],
  [5.0,3.6,1.4,0.2],
  [5.4,3.9,1.7,0.4],
  [4.6,3.4,1.4,0.3],
  [5.0,3.4,1.5,0.2],
  [4.4,2.9,1.4,0.2],
  [4.9,3.1,1.5,0.1],
  [5.4,3.7,1.5,0.2],
  [4.8,3.4,1.6,0.2],
  [4.8,3.0,1.4,0.1],
  [4.3,3.0,1.1,0.1],
  [5.8,4.0,1.2,0.2],
  [5.7,4.4,1.5,0.4],
  [5.4,3.9,1.3,0.4],
  [5.1,3.5,1.4,0.3],
  [5.7,3.8,1.7,0.3],
  [5.1,3.8,1.5,0.3],
  [5.4,3.4,1.7,0.2],
  [5.1,3.7,1.5,0.4],
  [4.6,3.6,1.0,0.2],
  [5.1,3.3,1.7,0.5],
  [4.8,3.4,1.9,0.2],
  [5.0,3.0,1.6,0.2],
  [5.0,3.4,1.6,0.4],
  [5.2,3.5,1.5,0.2],
  [5.2,3.4,1.4,0.2],
  [4.7,3.2,1.6,0.2],
  [4.8,3.1,1.6,0.2],
  [5.4,3.4,1.5,0.4],
  [5.2,4.1,1.5,0.1],
  [5.5,4.2,1.4,0.2],
  [4.9,3.1,1.5,0.1],
  [5.0,3.2,1.2,0.2],
  [5.5,3.5,1.3,0.2],
  [4.9,3.1,1.5,0.1],
  [4.4,3.0,1.3,0.2],
  [5.1,3.4,1.5,0.2],
  [5.0,3.5,1.3,0.3],
  [4.5,2.3,1.3,0.3],
  [4.4,3.2,1.3,0.2],
  [5.0,3.5,1.6,0.6],
  [5.1,3.8,1.9,0.4],
  [4.8,3.0,1.4,0.3],
  [5.1,3.8,1.6,0.2],
  [4.6,3.2,1.4,0.2],
  [5.3,3.7,1.5,0.2],
  [5.0,3.3,1.4,0.2],
  [7.0,3.2,4.7,1.4],
  [6.4,3.2,4.5,1.5],
  [6.9,3.1,4.9,1.5],
  [5.5,2.3,4.0,1.3],
  [6.5,2.8,4.6,1.5],
  [5.7,2.8,4.5,1.3],
  [6.3,3.3,4.7,1.6],
  [4.9,2.4,3.3,1.0],
  [6.6,2.9,4.6,1.3],
  [5.2,2.7,3.9,1.4],
  [5.0,2.0,3.5,1.0],
  [5.9,3.0,4.2,1.5],
  [6.0,2.2,4.0,1.0],
  [6.1,2.9,4.7,1.4],
  [5.6,2.9,3.6,1.3],
  [6.7,3.1,4.4,1.4],
  [5.6,3.0,4.5,1.5],
  [5.8,2.7,4.1,1.0],
  [6.2,2.2,4.5,1.5],
  [5.6,2.5,3.9,1.1],
  [5.9,3.2,4.8,1.8],
  [6.1,2.8,4.0,1.3],
  [6.3,2.5,4.9,1.5],
  [6.1,2.8,4.7,1.2],
  [6.4,2.9,4.3,1.3],
  [6.6,3.0,4.4,1.4],
  [6.8,2.8,4.8,1.4],
  [6.7,3.0,5.0,1.7],
  [6.0,2.9,4.5,1.5],
  [5.7,2.6,3.5,1.0],
  [5.5,2.4,3.8,1.1],
  [5.5,2.4,3.7,1.0],
  [5.8,2.7,3.9,1.2],
  [6.0,2.7,5.1,1.6],
  [5.4,3.0,4.5,1.5],
  [6.0,3.4,4.5,1.6],
  [6.7,3.1,4.7,1.5],
  [6.3,2.3,4.4,1.3],
  [5.6,3.0,4.1,1.3],
  [5.5,2.5,4.0,1.3],
  [5.5,2.6,4.4,1.2],
  [6.1,3.0,4.6,1.4],
  [5.8,2.6,4.0,1.2],
  [5.0,2.3,3.3,1.0],
  [5.6,2.7,4.2,1.3],
  [5.7,3.0,4.2,1.2],
  [5.7,2.9,4.2,1.3],
  [6.2,2.9,4.3,1.3],
  [5.1,2.5,3.0,1.1],
  [5.7,2.8,4.1,1.3],
  [6.3,3.3,6.0,2.5],
  [5.8,2.7,5.1,1.9],
  [7.1,3.0,5.9,2.1],
  [6.3,2.9,5.6,1.8],
  [6.5,3.0,5.8,2.2],
  [7.6,3.0,6.6,2.1],
  [4.9,2.5,4.5,1.7],
  [7.3,2.9,6.3,1.8],
  [6.7,2.5,5.8,1.8],
  [7.2,3.6,6.1,2.5],
  [6.5,3.2,5.1,2.0],
  [6.4,2.7,5.3,1.9],
  [6.8,3.0,5.5,2.1],
  [5.7,2.5,5.0,2.0],
  [5.8,2.8,5.1,2.4],
  [6.4,3.2,5.3,2.3],
  [6.5,3.0,5.5,1.8],
  [7.7,3.8,6.7,2.2],
  [7.7,2.6,6.9,2.3],
  [6.0,2.2,5.0,1.5],
  [6.9,3.2,5.7,2.3],
  [5.6,2.8,4.9,2.0],
  [7.7,2.8,6.7,2.0],
  [6.3,2.7,4.9,1.8],
  [6.7,3.3,5.7,2.1],
  [7.2,3.2,6.0,1.8],
  [6.2,2.8,4.8,1.8],
  [6.1,3.0,4.9,1.8],
  [6.4,2.8,5.6,2.1],
  [7.2,3.0,5.8,1.6],
  [7.4,2.8,6.1,1.9],
  [7.9,3.8,6.4,2.0],
  [6.4,2.8,5.6,2.2],
  [6.3,2.8,5.1,1.5],
  [6.1,2.6,5.6,1.4],
  [7.7,3.0,6.1,2.3],
  [6.3,3.4,5.6,2.4],
  [6.4,3.1,5.5,1.8],
  [6.0,3.0,4.8,1.8],
  [6.9,3.1,5.4,2.1],
  [6.7,3.1,5.6,2.4],
  [6.9,3.1,5.1,2.3],
  [5.8,2.7,5.1,1.9],
  [6.8,3.2,5.9,2.3],
  [6.7,3.3,5.7,2.5],
  [6.7,3.0,5.2,2.3],
  [6.3,2.5,5.0,1.9],
  [6.5,3.0,5.2,2.0],
  [6.2,3.4,5.4,2.3],
  [5.9,3.0,5.1,1.8],
])

IRIS_LABELS = [
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
]

def test_typicality():
  with open("cache/cached-features.pkl", 'rb') as fin:
    features = pickle.load(fin)
  with open("cache/cached-projection.pkl", 'rb') as fin:
    proj = pickle.load(fin)
  for points, projected in [(features, proj)]:
    typ = multiscale.typicality(points, quiet=False)
    plot_property(projected, typ, title="Typicality")
    plt.show()

def test_representatives():
  for t in test_cases:
    tc = test_cases[t]
    reps = multiscale.find_representatives(tc)
    plot_reps(tc, reps)
    plt.show()

def project(x):
  return x[:,:2]

def get_stuff(seed, which="cases", target_projected=False, subset=None):
  if which == "iris":
    stuff = [("iris", IRIS_DATA, project(IRIS_DATA))]
  elif which == "features":
    features = utils.load_cache("features")
    proj = utils.load_cache("projected")
    stuff = [("features", features, proj)]
  else:
    if subset:
      items = [(n, t) for (n, t) in test_cases.items() if n in subset]
    else:
      items = test_cases.items()
    keys = [it[0] for it in items]
    values = [it[1] for it in items]
    stuff = zip(keys, values, [project(it[1]) for it in items])

  if target_projected:
    stuff = [(a + "-projected", c, c) for (a, b, c) in stuff]

  stuff = [(a + "%" + str(seed), b, c) for (a, b, c) in stuff]

  return stuff

def get_sparse_edges(name, points, fresh=False):
  print("Finding nearest neighbors to generate edge set...")
  nbd, nbi = utils.cached_values(
    lambda: multiscale.neighbors_from_points(points, 20, "euclidean"),
    (
      "multiscale-neighbor-distances-20-{}".format(name),
      "multiscale-neighbor-indices-20-{}".format(name),
    ),
    ("pkl", "pkl"),
    override=fresh,
    debug=print
  )
  print("...done.")
  print("Generating edges from nearest neighbors...")
  edges = utils.cached_value(
    lambda: multiscale.get_neighbor_edges(nbd, nbi),
    "edges-{}".format(name),
    override=fresh,
    debug=print
  )
  print("...done.")
  return edges

def check_edges(points, edges, epsilon=0.0000000001):
  for (fr, to, d) in edges:
    rd = euclidean(points[fr], points[to])
    if abs(rd - d) > epsilon:
      print(
        "Found incongruent edge {} -> {} ({} != {})".format(fr, to, d, rd),
        file=sys.stderr
      )
      return False
  return True

def test(seed, fresh, which, subset):
  #target_projected=True
  target_projected=False

  #stuff = get_stuff(seed, which, target_projected)
  stuff = get_stuff(
    seed,
    which,
    target_projected,
    subset=subset
  )

  for (name, points, projected) in stuff:
    edges = get_sparse_edges(name, points, fresh)
    clusters = multiscale.multiscale_clusters(
      points,
      edges=edges,
      neighbors_cache_name="multiscale-neighbors-{}".format(name),
      cache_neighbors=not fresh,
      quiet=False,
    )

    #root = multiscale.find_largest(clusters)
    #plot_clusters(
    #  name,
    #  points,
    #  {0: root},
    #  title="Root Cluster",
    #  show_quality="edges"
    #)

    top = clusters

    #print("Retaining best clusters...")
    #top = multiscale.retain_best(top, filter_on="quality")
    #print("...retained {} clusters.".format(len(top)))

    #print("Retaining best clusters...")
    #top = multiscale.retain_best(top, filter_on="mixed_quality")
    #print("...retained {} best clusters.".format(len(top)))
    #print("Sifting child clusters...")
    #top = multiscale.sift_children(
    #  top,
    #  threshold=1.0,
    #  quality="mixed_quality"
    #)
    #print("...found {} sifted clusters.".format(len(top)))
    print("Retaining only large clusters...")
    top = multiscale.retain_above(
      top,
      threshold=10,
      filter_on="size"
    )
    print("...found {} large clusters.".format(len(top)))
    #top = multiscale.retain_best(top, filter_on="mixed_quality")
    #top = multiscale.retain_best(clusters, filter_on="compactness")
    #top = multiscale.retain_best(top, filter_on="separation")

    #print("Splitting best clusters...")
    #sep = multiscale.decant_split(
    #  top,
    #  threshold=1.0,
    #  #criterion=multiscale.quality_vs_coverage_criterion(
    #  #  size="core_size",
    #  #  quality="quality"
    #  #)
    #  #criterion=lambda cl: cl["size"] / (len(points)/4)
    #  #threshold=0.9,
    #  criterion=multiscale.generational_criterion(feature="adjusted_mixed")
    #  #criterion=multiscale.generational_criterion(feature="separation")
    #  #criterion=multiscale.product_criterion(
    #  #  multiscale.generational_criterion(feature="separation"),
    #  #  multiscale.generational_criterion(feature="quality")
    #  #)
    #  #criterion=multiscale.quality_vs_outliers_criterion(quality="quality")
    #  #criterion=multiscale.satisfaction_criterion(quality="quality")
    #  #criterion=multiscale.scale_quality_criterion(quality="quality")
    #)
    #print("...retained {} best clusters.".format(len(sep)))

    print("Condensing best clusters...")
    sep = multiscale.condense_best(
      top,
      discriminant="mean",
      quality="adjusted_mixed",
      distinction=0.3,
      tolerance=0.8
    )
    print("...retained {} best clusters.".format(len(sep)))

    #print("Eroding best clusters...")
    #sep = multiscale.decant_erode(points, top, threshold=0.6)
    #print("...retained {} best clusters.".format(len(sep)))

    #print("Electing best clusters...")
    #sep = multiscale.vote_refine(points, top, quality="adjusted_mixed")
    #sep = multiscale.vote_refine(points, top, quality="mixed_quality")
    #print("...retained {} best clusters.".format(len(sep)))

    #utils.reset_color()
    #plot_cluster_sequence(projected, clusters)

    #utils.reset_color()
    #plot_clusters(
    #  projected,
    #  clusters,
    #  title="All {} Clusters".format(len(clusters))
    #)

    #utils.reset_color()
    #plot_clusters(
    #  projected,
    #  top,
    #  title="{} filtered ({} cluster(s))".format(name, len(top))
    #)

    utils.reset_color()
    plot_clusters(
      projected,
      sep,
      title="{} condensed ({} cluster(s))".format(name, len(sep))
    )

    #utils.reset_color()
    #plot_clusters(
    #  projected,
    #  best,
    #  title="{} revised ({} cluster(s))".format(name, len(best))
    #)

    which_stats = [
      "mean",
      "size",
      "scale",
      #"quality",
      #"obviousness",
      "coherence",
      "dominance",
      #"compactness",
      #"separation",
      #"mixed_quality",
      "adjusted_mixed",
      #"split_quality",
    ]

    stats_sort = "scale"
    #stats_sort = "size"
    #stats_sort = "mean"

    stats_norm = ["mean", "size", "scale"]

    stats_means = ["adjusted_mixed"]

    #utils.reset_color()
    #plot_stats(
    #  name,
    #  clusters,
    #  which_stats,
    #  sort_by=stats_sort,
    #  normalize=stats_norm,
    #  show_mean=stats_means,
    #)

    #utils.reset_color()
    #plot_stats(
    #  name + "-top",
    #  top,
    #  which_stats,
    #  sort_by=stats_sort,
    #  normalize=stats_norm,
    #  show_mean=stats_means,
    #)

    #analyze_clustering(points, neighbors, clusters)

    plt.show()

def test_exact(seed, fresh, which, subset):
  #target_projected=True
  target_projected=False

  #stuff = get_stuff(seed, which, target_projected)
  stuff = get_stuff(
    seed,
    which,
    target_projected,
    subset=subset
  )

  for (name, points, projected) in stuff:
    edges = get_sparse_edges(name, points, fresh)
    clusters = multiscale.condensed_multiscale(
      points,
      edges=edges,
      neighbors_cache_name="multiscale-neighbors-{}".format(name),
      cache_neighbors=not fresh,
      quiet=False,
    )

    utils.reset_color()
    plot_clusters(
      projected,
      clusters,
      title="{} condensed ({} cluster(s))".format(name, len(clusters))
    )

    which_stats = [
      "mean",
      "size",
      "scale",
      #"quality",
      #"obviousness",
      "coherence",
      "dominance",
      #"compactness",
      #"separation",
      #"mixed_quality",
      "adjusted_mixed",
      #"split_quality",
    ]

    stats_sort = "scale"
    #stats_sort = "size"
    #stats_sort = "mean"

    stats_norm = ["mean", "size", "scale"]

    stats_means = []

    #utils.reset_color()
    #plot_stats(
    #  name,
    #  clusters,
    #  which_stats,
    #  sort_by=stats_sort,
    #  normalize=stats_norm,
    #  show_mean=stats_means,
    #)

    #analyze_clustering(points, neighbors, top)

    plt.show()

def test_anim(seed, fresh, which, subset):
  target_projected=True
  #target_projected=False

  stuff = get_stuff(
    seed,
    which,
    target_projected,
    subset=subset
  )

  for (name, points, projected) in stuff:
    edges = get_sparse_edges(name, points, fresh)
    if not check_edges(points, edges):
      print(
        "Edges are corrupted. Aborting. (try running with -F)",
        file=sys.stderr
      )
      exit(1)
    animate_mst(points, edges)

if __name__ == "__main__":
  fresh=False
  if "-F" in sys.argv:
    fresh=True

  seed = 17
  random.seed(seed)
  np.random.seed(seed)

  which = "cases"
  #which = "iris"
  #which = "features"

  #subset=[
  #  "quad_encircled",
  #  "overlapping",
  #  "complex",
  #  #"concentric",
  #  #"pair",
  #  #"scatter",
  #  #"normal",
  #  #"ring"
  #]
  subset = None

  #test(seed, fresh, which, subset)
  #utils.run_strict(test, seed, fresh, which, subset)
  #test_exact(seed, fresh, which, subset)
  #test_anim(seed, fresh, which, subset)
  #test_typicality()
  test_representatives()
