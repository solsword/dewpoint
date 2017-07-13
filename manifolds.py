"""
manifolds.py
Extracts clusters as manifolds and estimates their intrinsic dimensionality.
"""

import utils

# TODO: Clean up this file!

DEFAULT_MANIFOLD_DETECTION_PARAMETERS = {
  "improvement_threshold": 0.02,
  "approximation_samples": 5,
  "viz_grid_resolution": 10,
  "viz_edge_samples": 100,
}

def subclusters(points, neighbors, clustering):
  nbd, nbi = neighbors
  unions, edges, sig = clustering
  analyze = []
  for g in unions.groups():
    if len(g) >= params["useful_size"]:
      analyze.append(g)

  results = []
  for g in analyze:
    sc = {
      "points": [],
      "core_edges": [],
      "all_edges": [],
    }
    for pid in g:
      sc["points"].append(pid)
      for i, dst in enumerate(nbi[pid]):
        if dst in g:
          sc["all_edges"].append((pid, dst, nbd[pid,i]))

    anchor = unions.find(g[0])
    for fr, to, d, nr in edges:
      if unions.find(fr) == anchor:
        sc["core_edges"].append((fr, to, d))
        # ensure connectivity of all_edges graph:
        if (fr, to, d) not in sc["all_edges"]:
          sc["all_edges"].append((fr, to, d))

    results.append(sc)

  return results

def analyze_clustering(points, neighbors, clustering):
  subcl = subclusters(points, neighbors, clustering)
  if len(subcl) == 0:
    debug("No interesting subclusters to analyze.")
  else:
    for cluster in subcl:
      manifold = approximate_manifold(points, cluster)
      mdim = 1 + (len(manifold["anchors"]) - 2) / 2
      debug("Cluster size: {}".format(len(cluster["points"])))
      debug("Manifold edges: {}".format(len(cluster["all_edges"])))
      debug("Manifold dimensions: {}".format(mdim))
      plot_manifold(manifold, show_interior=False, show_edges=True)
    #plt.show()

def plot_manifold(
  manifold,
  show_interior=False,
  show_grid=False,
  show_edges=False
):
    n = manifold["exterior"].shape[0]
    nanchors = len(manifold["anchors"])
    mdim = 1 + (nanchors - 2) / 2
    fig, ax = plt.subplots()
    plt.title("Detected Manifold ({}-dimensional)".format(mdim))
    plt.axis("equal")

    # Manifold edges:
    earray = []
    for fr, to, d in manifold["edges"]:
      earray.append((manifold["exterior"][fr], manifold["exterior"][to]))
    earray = np.asarray(earray)
    lc = mc.LineCollection(
      earray,
      colors=(0.8, 0.8, 0.8),
      linewidths=0.7
    )
    ax.add_collection(lc)

    # Original points:
    plt.scatter(
      manifold["exterior"][:,0],
      manifold["exterior"][:,1],
      s=1.0,
      c=[(0.9, 0.9, 0.9)]*n
    )

    # Anchor points:
    apoints = manifold["exterior"][manifold["anchors"],:]
    plt.scatter(
      apoints[:,0],
      apoints[:,1],
      s=5.0,
      c=[(1.0, 0.0, 0.0)]*apoints.shape[0]
    )

    # Reconstructed points:
    plt.scatter(
      manifold["reconstructed"][:,0],
      manifold["reconstructed"][:,1],
      s=0.8,
      c=[(0.0, 0.4, 0.05)]*n
    )

    # Interior Points (at most 2 anchors-worth):
    if show_interior:
      if mdim < 1:
        plt.scatter(
          manifold["interior"][:,0],
          np.full(
            (manifold["interior"].shape[0],),
            np.mean(manifold["exterior"][:1])
          ),
          s=0.8,
          c=[(0.0, 0.1, 0.6)]*n
        )
      else:
        plt.scatter(
          manifold["interior"][:,0],
          manifold["interior"][:,1],
          s=0.8,
          c=[(0.4, 0.6, 1.0)]*n
        )

    # Grid:
    if show_grid:
      debug("Building manifold grid...")
      # A uniform grid over a unit simplex:
      gridpoints = simplex_grid(
        nanchors,
        MANIFOLD_GRID_RESOLUTION
      ) / MANIFOLD_GRID_RESOLUTION
      neighbors = NearestNeighbors(
        n_neighbors=nanchors,
        algorithm="ball_tree"
      ).fit(gridpoints)
      nbd, nbi = neighbors.kneighbors(gridpoints)
      gridedges = []
      for i in range(len(nbi)):
        for nb in nbi[i]:
          gridedges.append((i, nb))
      debug("  ...done building grid.")

      debug(
        "Translating grid into original space ({} points)...".format(
          len(gridpoints)
        )
      )
      extpoints = []
      for i, gp in enumerate(gridpoints):
        utils.prbar(i / len(gridpoints), debug=debug)
        extpoints.append(approximate_exterior_position(manifold, gp))
      extpoints = np.asarray(extpoints)

      extedges = [ (extpoints[i], extpoints[j]) for (i, j) in gridedges ]
      debug("  ...done translating grid.")

      plt.scatter(
        extpoints[:,0],
        extpoints[:,1],
        s=0.4,
        c=[(0.0, 0.0, 0.0)]*n
      )
      lc = mc.LineCollection(
        extedges,
        colors=(0.2, 0.2, 0.2),
        linewidths=0.7
      )
      ax.add_collection(lc)

    # Edges:
    if show_edges:
      debug("Building manifold edges...")
      extrema = np.asarray([
        [0]*i + [1] + [0]*(nanchors - i - 1)
          for i in range(nanchors)
      ])
      edges = []
      for i, fr in enumerate(extrema):
        for j, to in enumerate(extrema[i:]):
          edges.append(np.asarray([
            fr*(interp/MANIFOLD_EDGE_SAMPLES)
          + to * (MANIFOLD_EDGE_SAMPLES - interp)/MANIFOLD_EDGE_SAMPLES
              for interp in range(MANIFOLD_EDGE_SAMPLES+1)
          ]))
      debug("  ...done building edges.")


      debug("Translating edges into original space...")
      exedges = []
      for e in edges:
        ex = []
        for p in e:
          ex.append(approximate_exterior_position(manifold, p))
        exedges.append(ex)
      exedges = np.asarray(exedges)

      elen = exedges.shape[1]

      exlines = []
      for e in exedges:
        el = []
        for i in range(len(e)-1):
          el.append((e[i], e[i+1]))
        exlines.append(el)
      exlines = np.asarray(exlines)
      debug("  ...done translating edges.")

      for i in range(len(exedges)):
        plt.scatter(
          exedges[i][:,0],
          exedges[i][:,1],
          s=0.4,
          c=[(0.0, 0.0, 0.0)]*elen
        )
        lc = mc.LineCollection(
          exlines[i],
          colors=[(0.2, 0.2, 0.2)]*(elen-1),
          linewidths=0.7
        )
        ax.add_collection(lc)

def floyd_warshall(earray):
  # TODO: Get rid of this?
  n = earray.shape[0]
  for shorter in range(n):
    for fr in range(n):
      #if fr == shorter:
      #  continue
      for to in range(n):
        #if fr == to:
        #  continue
        better = earray[fr,shorter] + earray[shorter, to]
        if better < earray[fr, to]:
          earray[fr, to] = better

def collapse_cluster(points, cluster):
  result = {
    "count": 0,
    "scale": 0,
    "points": [],
    "edges": [],
  }
  pidmap = {}

  for pid in cluster["points"]:
    pidmap[pid] = len(result["points"])
    result["points"].append(points[pid])

  result["points"] = np.asarray(result["points"])
  result["count"] = result["points"].shape[0]

  result["scale"] = 0
  result["edges"] = []
  for fr, to, d in cluster["all_edges"]:
    result["scale"] += d
    result["edges"].append((pidmap[fr], pidmap[to], d))

  result["scale"] /= len(result["edges"])

  return result

# TODO: Relax points beforehand? Deal with noise better somehow!
@default_args(DEFAULT_MANIFOLD_DETECTION_PARAMETERS)
def approximate_manifold(points, cluster):
  # Find the shortest-path-distances-matrix for our cluster:
  local = collapse_cluster(points, cluster)

  longest_edge = max(d for fr, to, d in local["edges"])
  nonedge = longest_edge*local["count"]**2
  edgematrix = np.full((local["count"], local["count"]), nonedge)
  for fr, to, d in local["edges"]:
    edgematrix[fr,to] = d
    edgematrix[to,fr] = d

  for i in range(local["count"]):
    edgematrix[i,i] = 0

  shortest_paths = graph_shortest_path(edgematrix)
  if np.max(shortest_paths) == nonedge:
    raise RuntimeError("Cluster edges graph is not fully connected!")

  start = 0 # pick an arbitrary point
  extrema = [np.argmax(shortest_paths[start,:])]
  extrema.append(np.argmax(shortest_paths[extrema[0],:]))
  manifold_approximation = {
    "count": local["count"],
    "scale": local["scale"],
    "anchors": extrema,
    "exterior": local["points"],
    "interior": [],
    "reconstructed": [],
    "edges": local["edges"],
    "paths": shortest_paths,
  }
  compute_interior_positions(manifold_approximation)
  rerr = compute_reconstruction(manifold_approximation)
  prerr = rerr
  improvement = 1.0
  icount = 0
  debug(
    "Baseline reconstruction error ({} anchors): {:.5f}".format(
      len(manifold_approximation["anchors"]),
      rerr
    )
  )
  out_of_extrema = False
  while improvement > MANIFOLD_ERROR_THRESHOLD:
    icount += 1
    distances = manifold_approximation["paths"][
      manifold_approximation["anchors"],
      :
    ]
    combined = np.sum(distances**0.5, axis=0)**2
    bidx = np.argmax(combined)

    # TODO: Get rid of this debug code:
    #plot_manifold(manifold_approximation)
    #plt.show()
    if bidx in manifold_approximation["anchors"]:
      debug(
        "Warning: failed to find new extremum (found {} given {}).".format(
          bidx,
          manifold_approximation["anchors"],
        )
      )
      out_of_extrema = True
      break
    manifold_approximation["anchors"].append(bidx)
    compute_interior_positions(manifold_approximation)
    rerr = compute_reconstruction(manifold_approximation)
    improvement = prerr - rerr
    prerr = rerr
    debug(
      "[{}] Median reconstruction error ({} anchors): {:.5f} {:.3f}".format(
        icount,
        len(manifold_approximation["anchors"]),
        rerr,
        -improvement
      )
    )

  # Last anchor didn't improve things much, so remove it (not the case if we
  # ran out of extrema).
  if not out_of_extrema:
    manifold_approximation["anchors"] = manifold_approximation["anchors"][:-1]
    compute_interior_positions(manifold_approximation)
    compute_reconstruction(manifold_approximation)

  return manifold_approximation

def find_manifold_position(manifold, manifold_point_index):
  alines = manifold["paths"][manifold_point_index, manifold["anchors"]]
  return alines / np.sum(alines)

def compute_interior_positions(manifold):
  manifold["interior"] = []
  for mpi in range(manifold["count"]):
    manifold["interior"].append(find_manifold_position(manifold, mpi))
  manifold["interior"] = np.asarray(manifold["interior"])

def approximate_exterior_position(manifold, manifold_position):
  best = []
  for i in range(manifold["count"]):
    best.append(
      [i, params["metric"](manifold["interior"][i], manifold_position)]
    )
    best = sorted(best, key=lambda x: x[1])[:MANIFOLD_APPROXIMATION_SAMPLES]

  # invert & normalize distances:
  max_best = max(bd for (bi, bd) in best)
  second_best = min(bd for (bi, bd) in best if bd != 0)
  # TODO: Better way to avoid the singularity here?
  for i in range(len(best)):
      best[i][1] = (max_best + second_best) / (best[i][1] + second_best)
  sum_inv = sum(bs for (bi, bs) in best)
  for i in range(len(best)):
      best[i][1] /= sum_inv

  approx = np.zeros((manifold["exterior"].shape[1],))
  for bi, nbs in best:
    approx += manifold["exterior"][bi] * nbs
  return approx

def compute_reconstruction(manifold):
  errors = []
  manifold["reconstructed"] = []
  for i, ip in enumerate(manifold["interior"]):
    rp = approximate_exterior_position(manifold, ip)
    manifold["reconstructed"].append(rp)
    errors.append(params["metric"](rp, manifold["exterior"][i]) / manifold["scale"])

  manifold["reconstructed"] = np.asarray(manifold["reconstructed"])

  return np.median(errors)


