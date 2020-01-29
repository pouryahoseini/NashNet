#!/bin/bash

FILES=$(ls Configs/)

echo $FILES

for f in $FILES; do
  echo $f
done