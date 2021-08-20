#!/bin/bash
mkdir -p "../data/COMP66090-CW2/db"
curl https://dumps.wikimedia.org/enwiki/20210701/enwiki-20210701-pages-articles-multistream.xml.bz2 --output ../data/COMP66090-CW2/enwiki-20210701-pages-articles-multistream.xml.bz2
bzip2 -d ../data/COMP66090-CW2/enwiki-20210701-pages-articles-multistream.xml.bz2 ../data/COMP66090-CW2/wikidump/enwiki-20210701-pages-articles-multistream.xml