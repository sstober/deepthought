#!/bin/bash
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13
do
  echo "Starting $i... on gpu1"
  screen -dmS h0_worker$i ./startone.sh $i 1 1
done
