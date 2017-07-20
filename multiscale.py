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
from scipy.optimize import curve_fit

from simplex_grid.simplex_grid import simplex_grid

DEFAULT_CLUSTERING_PARAMETERS = {
  # Distance parameters:
  "distances": None,
  "edges": None,
  "metric": euclidean,
  "normalize_distances": False,
  "neighborhood_size": 4,
  # Cluster detection parameters:
  "quality_change_threshold": -0.1,
  "absolute_size_threshold": 6,
  "relative_size_threshold": 0.05,
  "significant_impact": 0.1,
  "minimum_impact_size": 2.5,
  "interest_threshold": 1.4,
  "linearity_window": 5,
  "outlier_criterion": 1.5,
  "cycle_limit": 3,
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
    "parent": None,
    "children": [A, B],
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
 #or c1["size"] < sizelimit * joined["size"]
 #or c2["size"] < sizelimit * joined["size"]
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


def get_distances(points, metric="euclidean", normalize=0):
  """
  Computes a distance matrix for the given points containing all pairwise
  distances using the given metric. If normalize is given, it should be an
  integer indicating the neighborhood size for distance normalization.
  """
  distances = pairwise.pairwise_distances(points, metric=metric)

  if normalize:
    neighbors = NearestNeighbors(
      n_neighbors=normalize,
      algorithm="ball_tree"
    ).fit(points)

    norm_distances, norm_indices = neighbors.kneighbors(points)
    local_scale = np.median(norm_distances, axis=1)
    normalized = np.zeros_like(distances)

    for i in range(distances.shape[0]):
      for j in range(distances.shape[1]):
        normalized[i,j] = distances[i,j] / ((local_scale[i] + local_scale[j])/2)

    return normalized

  else:
    return distances

def get_neighbor_edges(points, metric="euclidean", n=4):
  """
  Computes all edges between up to nth-nearest neighbors among the given
  points. Using these edges instead of edges from a full distance matrix can
  speed up clustering at the risk of failing to produce a minimum spanning
  tree.
  """
  neighbors = NearestNeighbors(
    n_neighbors=n,
    metric=metric,
    algorithm="ball_tree",
  ).fit(points)

  nbd, nbi = neighbors.kneighbors(points)

  return [
    (
      fr,
      nbi[fr][tidx],
      nbd[fr,tidx]
    )
      for fr in range(nbi.shape[0])
      for tidx in range(nbi.shape[1])
  ]

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
  n = len(points)
  debug = utils.get_debug(params["quiet"])
  debug("Starting clustering process...")
  if params["edges"] is None:
    debug("  ...computing edges...")
    if params["distances"] is None:
      debug("  ...computing pairwise distances...")
      distances = get_distances(
        points,
        metric=params["metric"],
        normalize=(
          params["neighborhood_size"]
            if params["normalize_distances"]
            else 0
        )
      )
      debug("  ...done.")
    else:
      debug("  ...using given pairwise distances...")
      distances = params["distances"]

    debug("  ...building edge list...")
    edges = []
    for fr in range(n):
      for to in range(fr+1, n):
        edges.append((fr, to, distances[fr,to]))
        if not params["symmetric"]:
          # if the metric is symmetric, no need to add this edge
          edges.append((to, fr, distances[to,fr]))

    edge_count = len(edges)
    debug("  ...found {} edges...".format(edge_count))
  else:
    edges = params["edges"]
    edge_count = len(edges)
    debug("  ...using {} given edges...".format(edge_count))

  debug("  ...sorting edges...")
  sorted_edges = sorted(edges, key=lambda e: (e[2], e[0], e[1]))
  debug("  ...done.")

  u = uf.unionfind(n)

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
        params["absolute_size_threshold"],
        params["relative_size_threshold"]
      )
    )

  debug("\n  ...done. Found {} clusters...".format(len(best)))
  debug("...done with clustering.")

  # Add normalized quality information:
  max_adj = 0
  max_coh = 0
  for k in best:
    cl = best[k]
    cl["adjusted_quality"] = cl["quality"] * math.log(cl["core_size"])
    cl["mixed_quality"] = (
      cl["quality"]
    #* cl["core_size"] / cl["size"]
    * math.log(cl["core_size"])
    )
    if max_adj < cl["adjusted_quality"]:
      max_adj = cl["adjusted_quality"]
    cl["coherence"] = cl["mean"] / cl["isolation_mean"]
    #cl["coherence"] = math.log(cl["core_size"]) / cl["isolation_mean"]
    if max_coh < cl["coherence"]:
      max_coh = cl["coherence"]

  for k in best:
    cl = best[k]
    cl["norm_quality"] = cl["adjusted_quality"] / max_adj
    cl["norm_coherence"] = cl["coherence"] / max_coh

  return best

def measure_deviation(
  clusters,
  stat="quality",
  against="size",
  fit="linear"
):
  """
  Adds a deviation measure to the given property by doing a regression of that
  property ordered by the given other property and measuring the difference
  between each point and the line. The regression is either linear (the
  default) or general exponential, according to the 'fit' parameter (which
  should be one of the strings "linear" or "exponential"). The added measure is
  named:

    <stat>_<against>_deviation
  """
  ordered = sorted(list(clusters.values()), key=lambda cl: cl[against])
  x = np.asarray([cl[against] for cl in ordered])
  y = np.asarray([cl[stat] for cl in ordered])

  if fit == "exponential":
    def expf(x, a, b, c):
      return a * np.exp(b*x) + c
    params, var = curve_fit(
      expf,
      x,
      y,
      p0=(0.5, -0.5, 0.5),
      bounds=((0, -1, 0), (1, 1, 1))
    )
    f = lambda x: expf(x, *params)
  elif fit == "linear":
    reg = linregress(x, y)
    f = lambda x: reg.intercept + reg.slope * x
  else:
    raise ValueError(
      (
        "Bad 'fit' parameter value '{}'"
        " (must be either 'linear' or 'exponential')."
      ).format(fit)
    )

  for k in clusters:
    cl = clusters[k]
    cl[stat + "_" + against + "_deviation"] = cl[stat] - f(cl[against])

def retain_best(clusters, filter_on="norm_quality"):
  """
  Filters a group of clusters on a particular property (in this case
  norm_quality) by retaining only clusters which exceed the mean value.

  TODO: don't mess with parent/children links!
  """
  mean = np.mean([clusters[k][filter_on] for k in clusters])
  filtered = {
    k: clusters[k]
      for k in clusters if clusters[k][filter_on] >= mean
  }

  for k in filtered:
    cl = filtered[k]
    cl["children"] = []
    while cl["parent"] != None and cl["parent"]["id"] not in filtered:
      cl["parent"] = cl["parent"]["parent"]

  for k in filtered:
    if filtered[k]["parent"]:
      filtered[k]["parent"]["children"].append(filtered[k])

  return filtered

def reassign_cluster_ids(clusters):
  newid = 0
  newgroup = {}
  for k in clusters:
    newgroup[newid] = clusters[k]
    clusters[k]["id"] = newid
    newid += 1

  return newgroup

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

def decant_split(clusters, threshold=1.0, criterion=lambda cl: 2.0):
  """
  Filters a group of clusters into a non-overlapping group, choosing to split
  up clusters according to the given criterion function and threshold (see some
  helpful examples below). If the criterion function returns a number higher
  than the threshold, the cluster is split.
  """
  results = {}
  assigned = set()

  srt = sorted(list(clusters.values()), key=lambda cl: -cl["size"])

  roots = []
  rooted = set()
  for cl in srt:
    if not cl["vertices"] & rooted:
      roots.append(cl)
      rooted |= cl["vertices"]

  clset = [ ]
  nextset = roots

  while len(nextset) > len(clset):
    clset = nextset
    nextset = []
    for cl in clset:
      if cl["size"] == 0 or len(cl["children"]) == 0: # can't split
        cl["split_quality"] = 0
        nextset.append(cl)
        continue
      cl["split_quality"] = criterion(cl)
      if cl["split_quality"] > threshold:
        nextset.extend(cl["children"])
      else:
        nextset.append(cl)

  # renumber the clusters:
  for i, cl in enumerate(clset):
    cl["id"] = i

  return { cl["id"]: cl for cl in clset }

def quality_vs_coverage_criterion(size="size", quality="quality"):
  """
  A splitting criterion for the decant_split function above. Compares the
  quality gained by splitting to the coverage (total size) lost. The given
  fields will be used as 'size' and 'quality' for the calculation. Pass the
  result of this function into decant_split.
  """
  def criterion(cluster):
    nonlocal size, quality
    child_coverage_ratio = (
      sum(child[size] for child in cluster["children"])
    / cluster[size]
    )
    child_quality_ratio = (
      max(child[quality] for child in cluster["children"])
    #  (
    #    sum(child[quality] for child in cluster["children"])
    #  / len(cluster["children"])
    #  )
    / cluster[quality]
    )
    return child_coverage_ratio * child_quality_ratio

  return criterion

def quality_vs_outliers_criterion(quality="quality"):
  """
  A splitting criterion for the decant_split function above. Compares the
  quality gained by splitting to the change in outlier percentage. The given
  fields will be used as the 'quality' for the calculation. Pass the result of
  this function into decant_split.
  """
  def criterion(cluster):
    nonlocal quality
    child_outliers_ratio = (
      (
        sum(child["size"] / child["core_size"] for child in cluster["children"])
      / len(cluster["children"])
      )
    / (cluster["size"] / cluster["core_size"])
    )
    child_quality_ratio = (
      max(child[quality] for child in cluster["children"])
    #  (
    #    sum(child[quality] for child in cluster["children"])
    #  / len(cluster["children"])
    #  )
    / cluster[quality]
    )
    return child_outliers_ratio * child_quality_ratio

  return criterion

def quality_criterion(quality="quality"):
  """
  A splitting criterion function that just uses quality.
  """
  def criterion(cluster):
    nonlocal quality
    return (
      (
        sum(child[quality] for child in cluster["children"])
      / len(cluster["children"])
      )
    / cluster[quality]
    )

  return criterion

def satisfaction_criterion(quality="quality"):
  """
  A splitting criterion function using size times quality.
  """
  def criterion(cluster):
    nonlocal quality
    satisfaction = cluster["core_size"] * cluster[quality]
    children_satisfaction = sum(
      child["core_size"] * child[quality] for child in cluster["children"]
    )
    return children_satisfaction / satisfaction

  return criterion

def scale_quality_criterion(quality="quality"):
  """
  A splitting criterion function using quality and scale gap.
  """
  def criterion(cluster):
    nonlocal quality
    child_quality_ratio = (
      (
        sum(child[quality] for child in cluster["children"])
      / len(cluster["children"])
      )
    / cluster[quality]
    )
    child_scale_ratio = (
      cluster["scale"]
    / (
        sum(child["scale"] for child in cluster["children"])
      / len(cluster["children"])
      )
    )
    return child_quality_ratio * child_scale_ratio

  return criterion

def vote_refine(
  all_points,
  clusters,
  leftovers_allowance=0.75,
  outlier_penalty=1.0,
  satisfaction_decay=0.8,
  quality="quality"
):
  """
  Refines a set of clusters by having every point vote for the best-quality
  cluster it's a member of and electing one cluster at a time until few
  unrepresented points (or no clusters) remain.
  """
  n = len(all_points)
  vote_weights = { p: 1.0 for p in range(n) }
  remaining_clusters = dict(clusters)
  elected = {}
  while (
    sum(vote_weights.values()) / n > leftovers_allowance
and remaining_clusters
  ):
    votes = {}
    for p in range(n):
      total_quality = 0
      my_votes = {}
      for k in remaining_clusters:
        cl = remaining_clusters[k]
        if p in cl["vertices"]:
          q = cl[quality]
          penalty = 1.0
          if p in cl["outliers"]:
            penalty = outlier_penalty
          my_votes[k] = q * penalty
          total_quality += q * penalty

      normalized = { k: my_votes[k]/total_quality for k in my_votes }
      for k in normalized:
        if k in votes:
          votes[k] += normalized[k] * vote_weights[p]
        else:
          votes[k] = normalized[k] * vote_weights[p]

    def quality_score(k):
      return votes[k] / clusters[k]["size"]

    best_cluster = None
    for k in votes:
      if best_cluster == None or quality_score(k) > quality_score(best_cluster):
        best_cluster = k

    print(
      "Elected {} (size {})".format(
        best_cluster,
        clusters[best_cluster]["size"]
      )
    )

    chosen = clusters[best_cluster]
    elected[best_cluster] = chosen
    for cp in chosen["vertices"]:
      vote_weights[cp] *= satisfaction_decay
    del remaining_clusters[best_cluster]

  return elected

@utils.default_params(DEFAULT_CLUSTERING_PARAMETERS)
def separated_multiscale(points, **params):
  """
  Calls multiscale_clusters with the given parameters and then filters the
  results a bit before returning them, using retain_best followed by
  decant_split to produce non-overlapping clusters.
  """
  initial_clusters = multiscale_clusters(points, **params)
  top = retain_best(initial_clusters, filter_on="mixed_quality")
  sep = decant_split(
    top,
    threshold=1.0,
    criterion=quality_vs_coverage_criterion(
      size="size",
      quality="quality"
    )
  )
  return sep

def cluster_assignments(points, clusters):
  clusters = reassign_cluster_ids(clusters)
  assignments = []
  for p in range(len(points)):
    assigned = False
    for k in clusters:
      if p in clusters[k]["vertices"]:
        assigned = True
        assignments.append(k)
        break
    if not assigned:
      assignments.append(-1)

  return assignments
