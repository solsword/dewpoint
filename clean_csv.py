#!/usr/bin/env python
"""
CSV cleanup script for tab-delimited but ambiguously-newlined files.
"""
import os
import sys
import csv
import re

CSV_FILE = os.path.join("data", "csv", "miiverse_profiles.final.csv")
OUTPUT_FILE = os.path.join("data", "csv", "miiverse_profiles.clean.csv")

if os.path.exists(OUTPUT_FILE):
  print(
    "Error: output file '{}' already exists. Aborting.".format(OUTPUT_FILE)
  )
  exit(1)

PR_INTRINSIC = 0
PR_CHARS = "▁▂▃▄▅▆▇█▇▆▅▄▃▂"
def prbar(progress):
  global PR_INTRINSIC
  pbwidth = 65
  sofar = int(pbwidth * progress)
  left = pbwidth - sofar - 1
  ic = PR_CHARS[PR_INTRINSIC]
  PR_INTRINSIC = (PR_INTRINSIC + 1) % len(PR_CHARS)
  print("\r[" + "="*sofar + ">" + "-"*left + "] (" + ic + ")", end="")

CSV_FIELDS = [
  "user-url",
  "avi-url",
  "country-code",
  "display-name",
  "username",
  "friends-with-limit",
  "following-raw",
  "followers-raw",
  "posts-raw",
  "yeahs-raw",
  "msg",
  "full-country",
  "birthdate",
  "competence",
  "platforms",
  "genres",
  "games"
]
EXTRACT = [
  "avi-url",
  "country-code",
  "frhttps://docs.python.org/3/library/csv.htmliends-with-limit",
  "following-raw",
  "followers-raw",
  "posts-raw",
  "yeahs-raw",
  "competence",
  "genres",
]
URL_RE = re.compile(r".*/([^/]+)_normal_face.png")
FRIENDS_RE = re.compile(r"(-|[0-9]+) / 100")
def firstgroup(expr):
  def cont(val):
    match = expr.match(val)
    if match:
      return match.group(1)
    else:
      return None
  return cont

def numeric(x):
  try:
    return int(x)
  except:
    return -1

PROCESS = {
  "avi-url": {
    "avi-id": firstgroup(URL_RE)
  },
  "friends-with-limit": {
    "friends":
      lambda with_limit: numeric(firstgroup(FRIENDS_RE)(with_limit))
  },
  "following-raw": {
    "following": numeric
  },
  "followers-raw": {
    "followers": numeric
  },
  "posts-raw": {
    "posts": numeric
  },
  "yeahs-raw": {
    "yeahs": numeric
  },
}
SAVE = {
  "avi-id",
  "country-code",
  "friends",
  "following",
  "followers",
  "posts",
  "yeahs",
  "competence",
  "genres",
}

def main():
  print("Reading CSV file...")
  with open(CSV_FILE, 'r') as fin:
    records = []
    here = []
    lines = fin.read()
    ll = len(lines)
    for i, line in enumerate(lines):
      prbar(i / ll)
      f = line.split('\t')
      if len(here):
        here[-1] += f[0]
        here.extend(f[1:])
      else:
        here = f

      if len(here) == len(CSV_FIELDS):
        here[-1] = here[-1][:-1] # trim newline from final field
        records.append(here)
        here = []
      elif len(here) > len(CSV_FIELDS):
        print(
          "Warning: line(s) with incorrect length {} (expected {}):".format(
            len(here),
            len(CSV_FIELDS)
          ),
          file=sys.stderr
        )
        print(here, file=sys.stderr)
        print("Ignoring unparsable line(s).", file=sys.stderr)
        here = []
  lr = len(records)
  print("\n  ...done ({} records extracted).".format(lr))

  save = [SAVE]
  print("Processing records...")
  for i, record in enumerate(records):
    prbar(i / lr)
    fields = {
      CSV_FIELDS[j]: record[j] for j in range(len(CSV_FIELDS))
    }
    extracted = {
      key: fields[key] for key in EXTRACT
    }
    for key in PROCESS:
      for new_key in PROCESS[key]:
        extracted[new_key] = PROCESS[key][new_key](extracted[key])
    save.append([extracted[key] for key in SAVE])
  print("\n  ...done.")

  if os.path.exists(OUTPUT_FILE):
    print(
      "Error: output file '{}' already exists. Aborting.".format(OUTPUT_FILE)
    )
    exit(1)

  print("Saving clean CSV...")
  with open(OUTPUT_FILE, 'w') as fout:
    writer = csv.writer(fout, dialect="excel")
    ls = len(save)
    for i, record in enumerate(save):
      prbar(i / ls)
      writer.writerow(record)
  print("\n  ...done.")

if __name__ == "__main__":
  main()
