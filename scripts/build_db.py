import pandas as pd
import sqlite3
import os
import glob
import json
import requests

from sqlalchemy import create_engine

path = '../data/FEVER/wiki-pages/'

engine = create_engine('sqlite:///../data/FEVER/db/wiki.db')

for filename in glob.glob(os.path.join(path, '*')):
    with open(filename, encoding='utf-8', mode='r') as currentFile:
        data = []
        print(currentFile)
        for line in currentFile.readlines()[1:]:
            doc = json.loads(line.replace('\n', ''))
            doc['title'] = doc['id']
            del doc['id']
            doc['title'] = doc['title'].replace('"', '')
            data.append(doc)
        
        df = pd.DataFrame(data)
        df.to_sql('CLAIMS_DOCUMENTS', engine, if_exists='append', index=False)
