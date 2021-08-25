import logging
import time
import os
import gensim
from gensim import models, utils, similarities

from gensim.test.utils import common_texts
from gensim.matutils import cossim
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import Word2Vec, TfidfModel
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex, SparseMatrixSimilarity

import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.tag import StanfordNERTagger
from nltk.chunk import conlltags2tree

import re
import json
import numpy as np
import ahocorasick
import tempfile

import spotlight

def dbpedia_annotations(sentence):
    
    doc_matches = []
    
    try:
        annotations = spotlight.annotate('https://api.dbpedia-spotlight.org/en/annotate', text=sentence, confidence=0.4, support=20, spotter='Default', disambiguator='Default', filters=None, headers=None)
    except:
        annotations = []

    annotations = list({v['URI']:v for v in annotations}.values())

    for annotation in annotations:
        title = annotation['URI'].replace('http://dbpedia.org/resource/', '').replace('_', ' ')
        surfaceForm = annotation['surfaceForm']
        try:
            doc_matches.append([titles_to_id[title], title])
        except:
            try:
                doc_matches.append([titles_to_id[surfaceForm], surfaceForm])
            except:
                pass
    
    print("DOC MATCHES:")
    print(doc_matches)
    
    return doc_matches

def stanford_entities(classified_text):
    
    entities = []
    
    ne_tree = stanfordNE2tree(classified_text)

    ne_in_sent = []
    for subtree in ne_tree:
        if type(subtree) == Tree: # If subtree is a noun chunk, i.e. NE != "O"
            ne_label = subtree.label()
            ne_string = " ".join([token for token, pos in subtree.leaves()])
            ne_in_sent.append((ne_string, ne_label))

    for ne in ne_in_sent:
        if ((len(ne[0]) > 4) and (ne[0] not in entities)):
            entities.append(ne[0])
            
    return entities

def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        #return "The strings are {} edits away".format(distance[row][col])
        return(distance[row][col])

def aho_corasick(entities, titles_to_id):

    auto = ahocorasick.Automaton()
    for substr in entities:
        auto.add_word(substr, substr)
    auto.make_automaton()

    w = tempfile.TemporaryFile()

    for astr in titles_to_id:
        for end_ind, found in auto.iter(astr):
            #print(found)
            w.write((found+astr).encode())
            #w.write(found+astr)

    search_space = []

    x = 0

    for astr in titles_to_id:
        seen = set()
        for end_ind, found in auto.iter(astr):
            if found not in seen:
                search_space.append([x, astr])
                seen.add(found)
                w.write((found+astr).encode())
                #w.write(found+astr)
        x = x + 1

    return search_space

def substring(entities, doc_matches):

    auto = ahocorasick.Automaton()
    for substr in entities:
        auto.add_word(substr, substr)
    auto.make_automaton()
    
    remove_entities = []
        
    for astr in doc_matches:
        for found in auto.iter(astr[1]):
            remove_entities.append(found[1])
            
    return remove_entities
    

# def formatted_entities(classified_paragraphs_list):
#     entities = {'persons': list(), 'organizations': list(), 'locations': list()}

#     for entry in classified_paragraphs_list:
#         entry_value = entry[0]
#         entry_type = entry[1]

#         if entry_type == 'PERSON':
#             entities['persons'].append(entry_value)

#         elif entry_type == 'ORGANIZATION':
#             entities['organizations'].append(entry_value)

#         elif entry_type == 'LOCATION':
#             entities['locations'].append(entry_value)

#     return entities

def formatted_entities(classified_paragraphs_list):
    entities = {'persons': list(), 'organizations': list(), 'locations': list()}

    i = 0
    while i < (len(classified_paragraphs_list)-1):
        
        entry = classified_paragraphs_list[i]
        
        entry_value = entry[0]
        entry_type = entry[1]

        # if entry_type == 'PERSON':
                        
        #     for j in range(i + 1, len(classified_paragraphs_list)-1):
        #         next_entry = classified_paragraphs_list[j]
        #         next_entry_value = next_entry[0]
        #         next_entry_type = next_entry[1]
                                
        #         if next_entry_type == 'PERSON':
        #             entry_value = entry_value + " " + next_entry_value
        #         else:
        #             break
                    
        #     entities['persons'].append(entry_value)
        #     i += (len(entry_value.split(' ')) - 1)

        # elif entry_type == 'ORGANIZATION':
        #     entities['organizations'].append(entry_value)

        # elif entry_type == 'LOCATION':
        #     entities['locations'].append(entry_value)
        
        #i += 1
        
        for j in range(i + 1, len(classified_paragraphs_list)-1):
            next_entry = classified_paragraphs_list[j]
            next_entry_value = next_entry[0]
            next_entry_type = next_entry[1]
                            
            if next_entry_type == entry_type:
                entry_value = entry_value + " " + next_entry_value
            else:
                break
        
        if entry_type == 'PERSON':       
            entities['persons'].append(entry_value)
            #i += (len(entry_value.split(' ')) - 1)

        elif entry_type == 'ORGANIZATION':
            entities['organizations'].append(entry_value)
            

        elif entry_type == 'LOCATION':
            entities['locations'].append(entry_value)
         
        i += len(entry_value.split(' '))
         

    return entities

def stanfordNE2BIO(tagged_sent):
    bio_tagged_sent = []
    prev_tag = "O"
    for token, tag in tagged_sent:
        if tag == "O": #O
            bio_tagged_sent.append((token, tag))
            prev_tag = tag
            continue
        if tag != "O" and prev_tag == "O": # Begin NE
            bio_tagged_sent.append((token, "B-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag == tag: # Inside NE
            bio_tagged_sent.append((token, "I-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag != tag: # Adjacent NE
            bio_tagged_sent.append((token, "B-"+tag))
            prev_tag = tag

    return bio_tagged_sent


def stanfordNE2tree(ne_tagged_sent):
    bio_tagged_sent = stanfordNE2BIO(ne_tagged_sent)
    sent_tokens, sent_ne_tags = zip(*bio_tagged_sent)
    sent_pos_tags = [pos for token, pos in pos_tag(sent_tokens)]

    sent_conlltags = [(token, pos, ne) for token, pos, ne in zip(sent_tokens, sent_pos_tags, sent_ne_tags)]
    ne_tree = conlltags2tree(sent_conlltags)
    return ne_tree

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load id->word mapping (the dictionary), one of the results of step 2 above
dictionary = gensim.corpora.Dictionary.load_from_text('corpus/_wordids.txt')
corpus = gensim.corpora.MmCorpus('corpus/_tfidf.mm')

st = StanfordNERTagger('stanford-ner-2020-11-17/classifiers/english.conll.4class.distsim.crf.ser.gz',
					   'stanford-ner-2020-11-17/stanford-ner.jar',
					   encoding='utf-8')

if (os.path.isfile('corpus/_titles_to_id.pickle')):
    titles_to_id = utils.unpickle('corpus/_titles_to_id.pickle')
    id_to_titles = utils.unpickle('corpus/_bow.mm.metadata.cpickle')
else:
    id_to_titles = utils.unpickle('corpus/_bow.mm.metadata.cpickle')

    # Create the reverse mapping, from article title to index.
    titles_to_id = {}

    # For each article...
    for at in id_to_titles.items():
        # `at` is (index, (pageid, article_title))  e.g., (0, ('12', 'Anarchism'))
        # at[1][1] is the article title.
        # The pagied property is unused.
        titles_to_id[at[1][1]] = at[0]

    # Store the resulting map.
    utils.pickle(titles_to_id, 'corpus/_titles_to_id.pickle')

f = open("extractedClaims.json", "r")

lines = f.readlines()

new_file = open("claimDocuments.json", "w")

total_claims = 0

for line in lines:
    js = json.loads(line)
    label = js['label']

    if (label == "CLAIM"):
        total_claims = total_claims + 1

print(total_claims)

claims = 1

for line in lines[11:]:
    js = json.loads(line)
    label = js['label']
    sentence = js['sentence']

    #doc_matches = []

    if (label == "CLAIM"):

        print(str(claims) + "/" + str(total_claims))

        print(sentence)

        claims = claims + 1
        
        doc_matches = dbpedia_annotations(sentence)

        #entities = []

        tokenized_text = word_tokenize(sentence)
        classified_text = st.tag(tokenized_text)
                
        entities = stanford_entities(classified_text)
        
        formatted_result = formatted_entities(classified_text)
        
        print(formatted_result)
        
        person_entites = formatted_result['persons']

        if (doc_matches != [] and entities != []):
            remove_entities = substring(entities, doc_matches)

            print("ENTITIES:")
            print(entities)
            print("ENTITIES TO REMOVE:")
            print(remove_entities)
            
            entities = list(set(entities) - set(remove_entities))
            person_entites = list(set(person_entites) - set(remove_entities))
            
            print("RESULT OF REMOVAL:")
            print(entities)

        sim_docs_scores = []
        
        search_space = []

        search_space = doc_matches

        if (entities != []):
            
            for person in person_entites:
                remove = False
                if (len(person.split(' ')) == 1):
                    per_search_space = aho_corasick([person], titles_to_id)
                    for doc in per_search_space:
                        if (re.match(r'([A-Z]{1}[a-z]+){1} ' + person + "$", doc[1])):
                            remove = True
                            entities.append(doc[1])
                            
                if (remove):
                    print(entities)
                    print(person)
                    entities.remove(person)

            for entity in entities:
                ent_search_space = aho_corasick([entity], titles_to_id)
                dists = []
                for doc in ent_search_space:
                    dist = levenshtein_ratio_and_distance(entity, doc[1])
                    dists.append(dist)

                idx = np.argsort(dists)

                dists = np.array(dists)[idx].tolist()
                ent_search_space = np.array(ent_search_space)[idx].tolist()

                search_space = search_space + ent_search_space#[:20]
                
        if (search_space != []):

            sim_scores = []
            sim_docs = []

            for doc in search_space:
                sent_1 = sentence
                sent_2 = doc[1]
                sim_scores.append(levenshtein_ratio_and_distance(sent_1, sent_2))
                sim_docs.append(id_to_titles[titles_to_id[doc[1]]])

            idx = np.argsort(sim_scores)

            sim_scores = np.array(sim_scores)[idx].tolist()
            sim_docs = np.array(sim_docs)[idx].tolist()

            for i in range(0, len(sim_scores)):
                sim_docs_scores.append(json.dumps({'id': sim_docs[i][0], 'title': sim_docs[i][1], 'distance': sim_scores[i]}))


        obj = json.dumps({'claim': sentence, 'docs': sim_docs_scores})

        print(obj)

        new_file.write(obj + "\n")

new_file.close()
