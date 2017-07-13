"""
multiscale.py
Implements a multi-scale minimum-spanning-tree-based clustering algorithm given
a distance matrix.
"""

# local imports:
import utils

import math

import numpy as np

import unionfind as uf

from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_shortest_path
from scipy.stats import linregress
from scipy.spatial.distance import euclidean

from simplex_grid.simplex_grid import simplex_grid

DEFAULT_CLUSTERING_PARAMETERS = {
  # Neighborhood sizes:
  "min_size": 3,
  "neighborhood_size": 4,
  "useful_size": 10,
  # Cluster detection parameters:
  "quality_change_threshold": -0.1,
  "threshold_adjust": 1.5,
  "significant_impact": 0.1,
  "minimum_impact_size": 2.5,
  "interest_threshold": 1.4,
  "linearity_window": 5,
  "outlier_criterion": 1.5,
  "cycle_limit": 3,
  "metric": euclidean,
  "symmetric": True,
  "quiet": True,
}

def join_clusters(A, B, e):
  """
  Combines two clusters being connected by an edge e, returning a new cluster
  representing the combination (the original clusters aren't modified). The
  quality change associated with the formation of this new cluster is also
  returned, and is added to the given edge as a fourth element.

  It incrementally maintains the cluster mean edge length as well as the
  internal-measure for the cluster, which is the sum of the similarities
  between each pair of points in the cluster, where the similarity between two
  points is defined by the maximum length of any edge on the (shortest) path
  between them (of course, in a minimum spanning tree there's only ever a
  single path between points).

  Notably, when two clusters whose points are all close to each other are
  joined by an edge that's long relative to their existing edges, the
  internal-measure of the resulting cluster is much larger than those of the
  clusters being combined. This effect is less pronounced when the edge joining
  the clusters is roughly the same length as edges already in the clusters,
  especially if we control for cluster size (see the cluster_quality function
  below).
  """

  fr, to, d = e
  sA = A["size"]
  sB = B["size"]
  mA = A["mean"]
  mB = B["mean"]

  nc = {
    "size": sA + sB + 1,
    "vertices": A["vertices"] | B["vertices"],
    "edges": A["edges"] | B["edges"], # new edge not included yet
    "mean": (mA * sA + mB * sB + d) / (sA + sB + 1),
    "internal": A["internal"] + B["internal"] + d * ((sA + 1) * (sB + 1)),
  }

  qA = cluster_quality(A)
  qB = cluster_quality(B)
  nq = cluster_quality(nc)

  pr_quality = ((sA + 1) * qA + (sB + 1) * qB) / (sA + sB + 2)
  qchange = nq - pr_quality

  # add the edge with its quality annotated
  nc["edges"] |= { (fr, to, d, qchange) }

  return nc, qchange

def cluster_quality(cluster):
  """
  Determine the quality of a cluster, according to the ratio between its
  internal-measure and a reference internal-measure (see the join_clusters
  function above).

  For clusters whose internal-measure is zero (i.e., clusters without edges or
  clusters composed entirely of duplicates) we set the quality to 1.0.

  The reference measure is the internal-measure for cluster of equivalent size
  where the largest edge on the (shortest) path between any two points in the
  cluster is equal to the mean edge length within the cluster. We compare that
  against the actual internal-measure for the cluster, which is just the sum of
  the maximum-length edge on the (shortest) path between every pair of points
  in the cluster.
  """
  i = cluster["internal"]
  if i == 0: # The cluster contains no non-duplicates
    return 1.0
  s = cluster["size"] + 1 # number of points rather than edges
  m = cluster["mean"]
  tp = (s * (s - 1)) / 2 # total point pairs in the cluster
  ref = m * tp # the reference value described above
  return ref / i

def merge_result(clusters, A, B, e):
  """
  Calculates the cluster that would result from joining clusters A and B using
  edge e. Mostly just a wrapper around join_clusters.
  """
  c1 = clusters[A]
  c2 = clusters[B]
  c1s = c1["size"]
  c2s = c2["size"]
  c1q = cluster_quality(c1)
  c2q = cluster_quality(c2)
  nc, qc = join_clusters(c1, c2, e)

  return nc, qc

def add_edge(unions, clusters, on_deck, edge, threshold):
  """
  Attempts to add the given edge to the given set of clusters (using the given
  unionfind data structure to detect cycle-inducing edges). If edge causes a
  change in quality below the given threshold (should usually be negative) then
  it will be added to the on_deck dictionary instead. If an edge passes the
  quality check but there's already another edge on deck to join the same two
  clusters, the waiting edge is used and the new edge is discarded (as it would
  now be a cycle-inducing edge). Returns True if an edge was added and False if
  it was discarded or put on deck.
  """
  (fr, to, d) = edge
  r1 = unions.find(fr)
  r2 = unions.find(to)

  if r1 == r2: # if this edge would create a cycle; ignore it
    return False

  j1, j2 = (min(r1, r2), max(r1, r2)) # make sure ordering is consistent

  joined, join_quality = merge_result(
    clusters,
    j1,
    j2,
    (fr, to, d)
  )

  if join_quality < threshold:
    # check if we already have something on deck:
    if j1 in on_deck:
      if j2 in on_deck[j1]:
        return False
    # if not, put this edge on deck before returning False:
    if j1 not in on_deck:
      on_deck[j1] = {}
    on_deck[j1][j2] = edge
    return False

  # if the quality is okay, check the deck for a better edge:
  if j1 in on_deck:
    if j2 in on_deck[j1]:
      joined, _ = merge_result(clusters, j1, j2, on_deck[j1][j2])

  # add the edge and combine the clusters it connects
  del clusters[j1]
  del clusters[j2]
  unions.unite(fr, to)
  nr = unions.find(fr)
  other = j1 if j1 != nr else j2
  clusters[nr] = joined

  # update the deck:

  # remove deck between just-joined clusters:
  if nr in on_deck:
    if other in on_deck[nr]:
      del on_deck[nr][other]

  # move items from defunct cluster to joined cluster, picking smaller edges
  # where there's a conflict:
  if other in on_deck:
    for to in on_deck[other]:
      if nr in on_deck:
        if to in on_deck[nr]:
          e1 = on_deck[other][to]
          e2 = on_deck[nr][to]
          best = e1 if e1[2] < e2[2] else e2
          on_deck[nr][to] = best
        else:
          on_deck[nr][to] = on_deck[other][to]
      else:
        on_deck[nr] = {}
        on_deck[nr][to] = on_deck[other][to]

    # defunct cluster no longer needs to be tracked:
    del on_deck[other]

  # redirect things from other clusters to the defunct cluster to point to the
  # merged cluster:
  for fr in on_deck:
    if other in on_deck[fr]:
      if nr in on_deck[fr]:
        e1 = on_deck[fr][other]
        e2 = on_deck[fr][nr]
        best = e1 if e1[2] < e2[2] else e2
        on_deck[fr][nr] = best
      else:
        on_deck[fr][nr] = on_deck[fr][other]

      # no need to keep this:
      del on_deck[fr][other]

  return True

@utils.default_params(DEFAULT_CLUSTERING_PARAMETERS)
def multiscale_clusters(points, **params):
  """
  This is the algorithm for computing multiscale clusters. It is a modified
  version of Kruskal's minimum-spanning-tree construction method which adds
  edges to a graph starting with the shortest edges and avoiding cycle-forming
  edges until a complete tree is formed.

  In this modified algorithm, edges which combine high-quality clusters into a
  low-quality cluster are skipped and revisited later, ultimately being thrown
  out if their quality-decreasing property persists. The result is a
  pseudo-minimum spanning forest of the data, which partitions it into
  clusters. Because only local distances are considered, clusters can have any
  shape.
  """
  debug = utils.get_debug(params["quiet"])
  debug("Starting clustering process...")
  debug("  ...computing pairwise distances...")
  distances = pairwise.pairwise_distances(points, metric=params["metric"])
  debug("  ...done.")

  n = distances.shape[0]
  edges = []
  for fr in range(n):
    for to in range(fr+1, n):
      edges.append((fr, to, distances[fr,to]))
      if not params["symmetric"]:
        # if the metric is symmetric, no need to add this edge
        edges.append((to, fr, distances[to,fr]))

  edge_count = len(edges)

  debug("  ...sorting edges...")
  sorted_edges = sorted(edges, key=lambda e: (e[2], e[0], e[1]))
  debug("  ...done.")

  u = uf.unionfind(distances.shape[0])

  debug("  ...creating point clusters...")
  clusters = {}
  for i in range(n):
    clusters[i] = {
      "size": 0,
      "vertices": { i },
      "edges": set(),
      "mean": 0,
      "internal": 0,
    }
  debug("  ...done.")
  debug("  ...constructing minimum spanning tree...")

  leftovers = sorted_edges

  cycle = 0
  added = 0
  while leftovers and cycle < params["cycle_limit"]:
    src = leftovers
    leftovers = []
    on_deck = {}
    debug("    ...scanning cycle {}...".format(cycle))
    for e in src:
      if added == n-1: # stop early
        break
      thr = (
        params["quality_change_threshold"]
      * params["threshold_adjust"]**cycle
      )
      utils.prbar(added / edge_count, debug=debug)

      # Add the next edge (possibly actually adding something from on deck, or
      # putting the edge on deck):
      added += int(add_edge(u, clusters, on_deck, e, thr))

    # Anything left on-deck is now left-over:
    for fr in on_deck:
      for to in on_deck[fr]:
        leftovers.append(on_deck[fr][to])

    cycle += 1 # done with this cycle
    debug(
      "\n    ...done with cycle (added {} edges)...".format(added)
    ) # done with progress bar too

  debug("  ...done. Found {} clusters...".format(len(clusters)))
  debug("...done with clustering.")

  assignment_set = list(set([u.find(i) for i in range(len(points))]))
  assignments = [
    assignment_set.index(u.find(i)) for i in range(len(points))
  ]

  return clusters, assignments
