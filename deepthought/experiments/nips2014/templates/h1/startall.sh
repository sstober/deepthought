#!/bin/bash
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13
do
  echo "Starting $i... on gpu0"
  screen -dmS h1_worker$i ./startone.sh $i 0 1
done
