#!/bin/bash
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13
do
  cleanup $i/
  rm -rf $i/output/*
  rm -rf $i/compile
done
