#!/bin/bash
mkdir -p "../data/json"
for WIKIDUMP in ../data/wikidumps/*
do
	firstString="../data/wikidumps/"
	secondString=""
	folder="${WIKIDUMP/$firstString/$secondString}"
	echo $folder
	mkdir -p "../data/json/$folder"
	python3 /media/sf_LEVEL_FOUR/MASTERS\ PROJECT/CW2/COMP66090-CW2/src/claimExtraction/wikiextractor/wikiextractor/WikiExtractor.py $WIKIDUMP -o ../data/json/$folder -b 3G --json --processes 10 -q
	rm -f $WIKIDUMP
done
rm -f ../data/wikidumps