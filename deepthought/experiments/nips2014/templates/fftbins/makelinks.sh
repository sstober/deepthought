#!/bin/bash
for bins in 1 3 6 9 12 15 18 21 24 28 32 36 40 44 49
do
	#subject_id=$(expr $i - 1)
	pyfile=$bins'_bins.py'
	mkdir $bins

	rm $bins/_base_config.properties
	echo 'n_freq_bins: '$bins > $bins/_base_config.properties
	#echo 'line 2' >> /tmp/newfile
	cat _base_config.properties >> $bins/_base_config.properties

	rm $bins/config.pb
	echo 'name: "'$bins'_bins"' > $bins/config.pb
        #echo 'line 2' >> /tmp/newfile
	cat config.pb >> $bins/config.pb

	rm $bins/$pyfile
	ln wrapper.py $bins/$pyfile

	rm $bins/_template.yaml
	ln _template.yaml $bins/_template.yaml
done
