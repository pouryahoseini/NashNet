#!/bin/bash

FILES=$(ls Configs/)

printf "Runnning using the following configuration files:\n$FILES\n"

mkdir Results

for f in $FILES; do
  echo "Running file $f"
  python3 Run.py "./Configs/$f"
  FNAME=$(basename "$f" | cut -d. -f1)
  mkdir -p "./Results/$FNAME/Model"
  mkdir -p "./Results/$FNAME/Reports"
  mv ./Model/*.json "./Results/$FNAME/Model"
  mv ./Model/*.h5 "./Results/$FNAME/Model"
  mv ./Reports/*.txt "./Results/$FNAME/Reports"
  mv ./Reports/*.csv "./Results/$FNAME/Reports"
done
