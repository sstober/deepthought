#!/bin/bash
for i in 18 21 24 28 32 36 40 44 49
#for i in 1
do
  echo "Starting $i... on gpu0"
  screen -dmS worker$i ./startone.sh $i 0 1
done

for i in 1 3 6 9 12 15
do
  echo "Starting $i... on gpu1"
  screen -dmS worker$i ./startone.sh $i 1 1
done
