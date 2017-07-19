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
  "absolute_size_threshold": 0.03,
  "relative_size_threshold": 0.08,
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

NEXT_CLUSTER_ID = 0

def join_clusters(A, B, e):
  """
  Combines two clusters being connected by an edge e, returning a new cluster
  representing the combination (the original clusters aren't modified). The
  quality of the new cluster is also returned, and is added to the given edge
  as a fourth element.

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
  especially if we control for cluster size (see the add_quality function
  below).
  """
  global NEXT_CLUSTER_ID

  fr, to, d = e
  sA = A["size"]
  sB = B["size"]
  mA = A["mean"]
  mB = B["mean"]

  combined_isolation = {}

  for v in A["isolation"]:
    avg, count = A["isolation"][v]
    combined_isolation[v] = (
      (avg * count + d * (sB + 1)) / (count + sB + 1),
      count + sB + 1
    )

  for v in B["isolation"]:
    avg, count = B["isolation"][v]
    combined_isolation[v] = (
      (avg * count + d * (sA + 1)) / (count + sA + 1),
      count + sA + 1
    )

  all_isolations = np.asarray([ic[0] for ic in combined_isolation.values()])
  isolation_mean = np.mean(all_isolations)
  isolation_std = np.std(all_isolations)

  outliers = {
    v
      for v in combined_isolation
      if combined_isolation[v][0] > isolation_mean + isolation_std
  }

  nc = {
    "id": NEXT_CLUSTER_ID,
    "size": sA + sB + 1,
    "scale": d,
    "vertices": A["vertices"] | B["vertices"],
    "isolation": combined_isolation,
    "isolation_mean": isolation_mean,
    "outliers": outliers,
    "core_size": sA + sB + 1 - len(outliers),
    "edges": A["edges"] | B["edges"], # new edge not included yet
    "mean": (mA * sA + mB * sB + d) / (sA + sB + 1),
    "internal": A["internal"] + B["internal"] + d * ((sA + 1) * (sB + 1)),
  }
  add_quality(nc)

  NEXT_CLUSTER_ID += 1

  # add the edge with its quality annotated
  nc["edges"] |= { (fr, to, d, nc["quality"]) }

  return nc

def add_quality(cluster):
  """
  Determine the quality of a cluster, according to the ratio between its
  internal-measure and a reference internal-measure (see the join_clusters
  function above), and add it to that cluster under the key "quality".

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
    cluster["quality"] = 1
    return

  s = cluster["size"] + 1 # number of points rather than edges
  m = cluster["mean"]
  tp = (s * (s - 1)) / 2 # total point pairs in the cluster
  ref = m * tp # the reference value described above
  softref = (m + np.std([d for (fr, to, d, q) in cluster["edges"]])) * tp
  cluster["quality"] = ref / i
  cluster["soft_quality"] = softref / i

def add_edge(unions, clusters, best, edge, minsize, sizelimit):
  """
  Attempts to add the given edge to the given set of clusters (using the given
  unionfind data structure to detect cycle-inducing edges). Maintains a
  dictionary of the best clusters found so far, merging best clusters when they
  remain good and preserving them when a merge decreases quality significantly.

  Returns True if an edge was added and False if it was discarded due to the
  no-cycles constraint.
  """
  (fr, to, d) = edge
  r1 = unions.find(fr)
  r2 = unions.find(to)
  c1 = clusters[r1]
  c2 = clusters[r2]

  if r1 == r2: # if this edge would create a cycle; ignore it
    return False

  joined = join_clusters(c1, c2, (fr, to, d))

  # add the edge and combine the clusters it connects
  del clusters[r1]
  del clusters[r2]
  unions.unite(fr, to)
  nr = unions.find(fr)
  clusters[nr] = joined

  joined["children"] = []

  if c1["size"] >= minsize:
    c1["parent"] = joined
    joined["children"].append(c1)

  if c2["size"] >= minsize:
    c2["parent"] = joined
    joined["children"].append(c2)

  # quality based subcluster memory:
  # update our best-clusters dictionary
  #if c1["id"] in best:
  #  if (joined["quality"] - best[c1["id"]]["quality"]) > threshold:
  #    del best[c1["id"]]
  #    joined["children"].remove(c1)
  #    for child in c1["children"]:
  #      child["parent"] = joined
  #      joined["children"].append(child)
  #    c1["parent"] = None
  #    best[joined["id"]] = joined

  #if c2["id"] in best:
  #  if (joined["quality"] - best[c2["id"]]["quality"]) > threshold:
  #    del best[c2["id"]]
  #    joined["children"].remove(c2)
  #    for child in c2["children"]:
  #      child["parent"] = joined
  #      joined["children"].append(child)
  #    c2["parent"] = None
  #    best[joined["id"]] = joined

  # size based subcluster memory:
  if (
    c1["size"] < minsize
 or c2["size"] < minsize
 or c1["size"] < sizelimit * joined["size"]
 or c2["size"] < sizelimit * joined["size"]
  ): 
    if c1["id"] in best:
      del best[c1["id"]]
    if c2["id"] in best:
      del best[c2["id"]]
    if c1 in joined["children"]:
      joined["children"].remove(c1)
    if c2 in joined["children"]:
      joined["children"].remove(c2)
    for child in c1["children"]:
      child["parent"] = joined
      joined["children"].append(child)
    for child in c2["children"]:
      child["parent"] = joined
      joined["children"].append(child)
    c1["parent"] = None
    c2["parent"] = None

  bi = sorted(list(best.items()), key=lambda kv: kv[1]["quality"])
  if bi:
    worst_id, worst = bi[0]

  if joined["size"] >= minsize:
    best[joined["id"]] = joined

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
  global NEXT_CLUSTER_ID
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
      "id": i,
      "parent": None,
      "children": [],
      "size": 0,
      "scale": 0,
      "vertices": { i },
      "isolation": { i: (0, 1) },
      "outliers": set(),
      "core_size": 1,
      "edges": set(),
      "mean": 0,
      "internal": 0,
      "quality": 1
    }

  NEXT_CLUSTER_ID = n

  debug("  ...done.")
  debug("  ...constructing minimum spanning tree...")

  added = 0
  src = sorted_edges
  best = {}
  for e in src:
    if added == n-1: # stop early
      break
    utils.prbar(added / edge_count, debug=debug)

    # Add the next edge (possibly actually adding something from on deck, or
    # putting the edge on deck):
    added += int(
      add_edge(
        u,
        clusters,
        best,
        e,
        #params["quality_change_threshold"],
        int(n * params["absolute_size_threshold"]),
        params["relative_size_threshold"]
      )
    )

  debug("  ...done. Found {} clusters...".format(len(best)))
  debug("...done with clustering.")

  # Add normalized quality information:
  max_adj = 0
  max_coh = 0
  for k in best:
    cl = best[k]
    cl["adjusted_quality"] = cl["quality"] * math.log(cl["core_size"])
    if max_adj < cl["adjusted_quality"]:
      max_adj = cl["adjusted_quality"]
    #cl["coherence"] = 1 / cl["isolation_mean"]
    cl["coherence"] = math.log(cl["core_size"]) / cl["isolation_mean"]
    if max_coh < cl["coherence"]:
      max_coh = cl["coherence"]

  for k in best:
    cl = best[k]
    cl["norm_quality"] = cl["adjusted_quality"] / max_adj
    cl["norm_coherence"] = cl["coherence"] / max_coh

  return best

def retain_best(clusters, filter_on="norm_quality"):
  """
  Filters a group of clusters on a particular property (in this case
  norm_quality) by retaining only clusters which exceed the mean value.
  """
  mean = np.mean([clusters[k][filter_on] for k in clusters])
  return {
    k: clusters[k]
      for k in clusters if clusters[k][filter_on] >= mean
  }

def decant_best(clusters, quality="quality"):
  """
  Filters a group of clusters into a non-overlapping group, by picking the best
  clusters first according to cluster quality and then ignoring clusters that
  would cause overlap.
  """
  results = {}
  assigned = set()

  srt = reversed(sorted(list(clusters.values()), key=lambda cl: cl[quality]))

  newid = 0
  for cl in srt:
    if cl["vertices"] & assigned:
      continue
    assigned |= cl["vertices"]
    cl["id"] = newid
    results[newid] = cl
    newid += 1

  return results

def decant_split(clusters, threshold=1.0, quality="quality"):
  """
  Filters a group of clusters into a non-overlapping group, choosing to split
  up larger clusters according to the balance between quality increase and
  coverage decrease that would result.
  """
  results = {}
  assigned = set()

  srt = sorted(list(clusters.values()), key=lambda cl: cl["size"])

  root = srt[-1]

  clset = [ ]
  nextset = [ root ]

  while len(nextset) > len(clset):
    clset = nextset
    nextset = []
    for cl in clset:
      if cl["size"] == 0 or len(cl["children"]) == 0: # can't split
        cl["split_quality"] = 0
        nextset.append(cl)
        continue
      child_coverage_ratio = (
        sum(child["core_size"] for child in cl["children"])
      / cl["core_size"]
      )
      child_quality_ratio = (
        max(child[quality] for child in cl["children"])
      #  (
      #    sum(child[quality] for child in cl["children"])
      #  / len(cl["children"])
      #  )
      / cl[quality]
      )
      cl["split_quality"] = child_coverage_ratio * child_quality_ratio
      if cl["split_quality"] > threshold:
        nextset.extend(cl["children"])
      else:
        nextset.append(cl)

  # renumber the clusters:
  for i, cl in enumerate(clset):
    cl["id"] = i

  return { cl["id"]: cl for cl in clset }
