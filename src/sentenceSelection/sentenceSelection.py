import sqlite3
import json
from nltk import tokenize

from gensim.test.utils import common_texts
from gensim.matutils import cossim
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex, SparseMatrixSimilarity

try:
    sqliteConnection = sqlite3.connect('data/db/wiki.db')
    cursor = sqliteConnection.cursor()
    print("Database created and Successfully Connected to SQLite")
    
    dictionary = Dictionary.load_from_text('../documentSelection/data/tfidf/_wordids.txt')
    corpus = MmCorpus('../documentSelection/data/tfidf/_tfidf.mm')

    model = TfidfModel.load('../documentSelection/data/tfidf/.tfidf_model')
    bow_corpus = MmCorpus('../documentSelection/data/tfidf/_bow.mm')

    docsim_index = SparseMatrixSimilarity(model[corpus], num_features=len(dictionary))

    new_file = open("output/claimSentences.json", "w")

    path = '../documentSelection/output/json/claimDocuments.json'

    claims = 1
    
    total_claims = len(open(path).readlines())

    with open(path, encoding='utf-8', mode='r') as currentFile:
        for line in currentFile:
            print(str(claims) + "/" + str(total_claims))
        
            claims_docs = json.loads(line)
            claim = claims_docs['claim']
            docs = claims_docs['docs']
            
            print(claim)
            
            big_sim = 0
            big_doc = {'title': None, 'text': None}
            big_sent = ""
            
            for doc in docs:
                doc = json.loads(doc)
                doc_id = doc['id']
                
                sqlite_select_Query = "SELECT text FROM CLAIMS_DOCUMENTS WHERE id = " + str(doc_id) + ";"
                
                cursor.execute(sqlite_select_Query)
                
                record = str(cursor.fetchall()[0])
                                                                
                for sentence in tokenize.sent_tokenize(record):
                    sent_1 = model[dictionary.doc2bow(claim.split())]
                    sent_2 = model[dictionary.doc2bow(sentence.split())]
                    
                    similarity = cossim(sent_1, sent_2)

                    if (similarity > big_sim):
                        big_sim = similarity
                        big_doc = doc
                        big_sent = sentence
                        
            obj = json.dumps({'claim': claim, 'sentence': big_sent})
        
            print(obj)

            new_file.write(obj + "\n")

        new_file.close()

            
            #print(claim + ":\n" + str(big_doc['title']) + " - " + str(big_sim) + "\n" + big_sent)

    sqlite_select_Query = "select sqlite_version();"
    cursor.execute(sqlite_select_Query)
    record = cursor.fetchall()
    print("SQLite Database Version is: ", record)
    cursor.close()

except sqlite3.Error as error:
    print("Error while connecting to sqlite", error)
finally:
    if sqliteConnection:
        sqliteConnection.close()
        print("The SQLite connection is closed")
        
