#!/bin/bash
mkdir export
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 'all' 'slow' 'fast'
do
	mkdir export/$i
	cp -Lrv $i/best* export/$i/
	cp $i/trace.csv export/$i/
done

