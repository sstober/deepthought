#!/bin/bash
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13
do
	subject_id=$(expr $i - 1)
	name='h1_subject'$i
	pyfile=$name'.py'
	mkdir $i

	rm $i/_base_config.properties
	echo 'subjects: ['$subject_id']' > $i/_base_config.properties
	cat _base_config.properties >> $i/_base_config.properties

	rm $i/config.pb
	echo 'name: "'$name'"' > $i/config.pb
	cat config.pb >> $i/config.pb

	rm $i/$pyfile
	ln wrapper.py $i/$pyfile

	rm $i/_template.yaml
	ln _template.yaml $i/_template.yaml
done
