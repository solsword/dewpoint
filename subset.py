#!/usr/bin/env python3
"""
subset.py

Get a random subset from a large data set.
"""

import os
import shutil
import sys
import random
import optparse

def main(parser, options, args):
  chance = options.use
  out = options.out

  if len(args) != 1:
    print(
      "Error: subset.py takes exactly 1 positional argument ({} given).".format(
        len(args)
      ),
      file=sys.stderr
    )
    parser.print_help(file=sys.stderr)
    exit(1)
  target = args[0]

  # Create output directory:
  try:
    os.mkdir(out, mode=0o755)
  except FileExistsError:
    print(
      "Warning: Output directory '{}' already exists.".format(out),
      file=sys.stderr
    )
    print(
      "Continue (this may overwrite files!) [y/N]?",
      end="",
      file=sys.stderr
    )
    inp = input()
    if inp not in ("Y", "y", "Yes", "yes"):
      print("Okay, aborting.", file=sys.stderr)
      exit(1)
    # else continue normally

  def copy(file):
    shutil.copy(
      file,
      os.path.join(
        out,
        os.path.relpath(file, target)
      )
    )

  n = 0
  seen = 0
  if options.extension:
    for root, dirs, files in os.walk(target):
      tdir = os.path.join(out, os.path.relpath(root, target))
      if not os.path.exists(tdir):
        os.mkdir(tdir, mode=0o755)
      for f in files:
        if f.endswith(options.extension):
          seen += 1
          if random.random() < chance:
            n += 1
            copy(os.path.join(root, f))
  else:
    for root, dirs, files in os.walk(target):
      tdir = os.path.join(out, os.path.relpath(root, target))
      if not os.path.exists(tdir):
        os.mkdir(tdir, mode=0o755)
      for f in files:
        if random.random() < chance:
          n += 1
          seen += 1
          copy(os.path.join(root, f))

  print("Copied {}/{} files from '{}' into '{}'.".format(n, seen, target, out))

if __name__ == "__main__":
  parser = optparse.OptionParser(
    usage="usage: %prog [options] TARGET\n" +\
          "  Copies a random subset of files from the TARGET directory."
  )
  parser.add_option(
    "-o",
    "--out",
    type="string",
    default="subset",
    help="Directory to write results into."
  )
  parser.add_option(
    "-e",
    "--extension",
    type="string",
    default=None,
    help="Required file extension. All other files will be ignored."
  )
  parser.add_option(
    "-u",
    "--use",
    type="float",
    default=0.1,
    help="Fraction of full dataset to use. Default 0.1."
  )
  options, args = parser.parse_args()
  main(parser, options, args)
