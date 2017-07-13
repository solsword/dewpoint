#!/bin/sh
mkdir -p data/mii_flat/
find data/mii_data/ -name "*.png" -exec convert "{}" -background magenta -alpha remove {}.flat.png \;
