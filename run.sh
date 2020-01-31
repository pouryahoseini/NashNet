#!/bin/bash

FILES=$(ls Configs/)

echo "Runnning using the following configuration files: $FILES"

mkdir Results

for f in $FILES; do
  echo "Running file $f"
  python3 Run.py $f
  FNAME=$(basename "$f" | cut -d. -f1)
  mkdir -p "./Results/$FNAME/Model"
  mkdir -p "./Results/$FNAME/Reports"
  mv ./Model/* "./Results/$FNAME/Model"
  mv ./Reports/* "./Results/$FNAME/Reports"
done