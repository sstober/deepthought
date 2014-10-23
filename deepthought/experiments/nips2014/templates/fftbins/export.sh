#!/bin/bash
mkdir export
for bins in 1 3 6 9 12 15 18 21 24 28 32 36 40 44 49
do
	mkdir export/$bins
	cp $bins/best* export/$bins/
	cp $bins/trace.csv export/$bins/
done

