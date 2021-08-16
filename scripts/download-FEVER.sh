#!/bin/bash
mkdir -p "../data/FEVER/db"
curl https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip --output ../data/FEVER/wiki-pages.zip
unzip ../data/FEVER/wiki-pages.zip -d ../data/FEVER/
python3 build_db.py
