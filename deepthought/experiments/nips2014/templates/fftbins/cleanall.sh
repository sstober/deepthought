#!/bin/bash
for i in 1 3 6 9 12 15 18 21 24 28 32 36 40 44 49
do
  cleanup $i/
  rm -rf $i/output/*
  rm -rf $i/compile
done
