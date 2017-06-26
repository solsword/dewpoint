#!/bin/sh
dir=$1
montage -geometry 128x128+2+2 "$dir"/best-image-* "$dir"/best-montage.png
montage -geometry 128x128+2+2 "$dir"/rec-image-* "$dir"/rec-montage.png
montage -geometry +4+4 "$dir"/*-montage.png "$dir"/combined.png
