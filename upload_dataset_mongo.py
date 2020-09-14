import os
import json
from pymongo import MongoClient

client = MongoClient()
db = client['sts']
collection = db['runs']
for root,directories,filenames in os.walk('./sts_runs'):
    print(root)
    for filename in filenames:
        try:
            with open(os.path.join(os.getcwd(),'sts_runs','ascension',filename),'r') as g:
                run = json.load(g)
            print(collection.insert_one(run))
        except UnicodeDecodeError:
            continue