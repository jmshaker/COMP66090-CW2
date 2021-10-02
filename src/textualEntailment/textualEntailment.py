import wandb
import torch
import gc
import json
import random
import numpy as np
import sys
sys.path.append('.')

from transformers import BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AdamW, DistilBertForSequenceClassification, RobertaForSequenceClassification
#from datasets import load_dataset, load_metric, Features

import sqlite3
import requests
from nltk import tokenize

sqliteConnection = sqlite3.connect('../data/db/wiki.db')
#sqliteConnection = sqlite3.connect('../data/FEVER/db/wiki.db')
cursor = sqliteConnection.cursor()
print("Database created and Successfully Connected to SQLite")

def getWikiCurid(title):
    url = "https://en.wikipedia.org/w/api.php?action=query&format=json&titles=" + title
    response = requests.get(url)
    obj = json.loads(response.text)['query']['pages']
    curid = int(list(obj.keys())[0])

    return curid

def getWikiSentences(title):
    try:
        sqlite_select_Query = "SELECT title, text, lines FROM CLAIMS_DOCUMENTS WHERE title = \"" + str(title) + "\";"

        cursor.execute(sqlite_select_Query)
        sqliteConnection.commit()

        record = cursor.fetchall()[0]

        title = record[0]
        text = record[1]
        lines = record[2]

        sentences = lines.splitlines()

        line_ids = []

        for sent in sentences:
            split = sent.split('\t')
            line_ids.append({"id": int(split[0]), "line": split[1]})

        return line_ids

    except:
        return []

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(predictions)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

from sentence_transformers import CrossEncoder
#model = CrossEncoder('cross-encoder/nli-roberta-base')
#model = CrossEncoder('cross-encoder/nli-distilroberta-base')

max_length = 256

hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
# hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
# hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
# hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
# hg_model_hub_name = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"

tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)

#Convert scores to labels
label_mapping = ['REFUTES', 'SUPPORTS', 'neutral']

new_file = open("../manuallyVerifiedClaims.jsonl", "r")

#new_file = open("../train.jsonl", "r")

claims = new_file.readlines()

print(claims)

for claim in claims:
    obj = json.loads(claim)
    print(obj)
    if (obj['verifiable'] == 'VERIFIABLE'):
        print(obj['claim'])
        evidence = obj['evidence']
        sentences = []
        for ev in evidence:
            sentences.append(ev)

        for sentence in sentences:
            #curid = getWikiCurid(sentence[0][2])
            #wikiSentences = getWikiSentences(curid)
            #print(wikiSentences)
            #print(sentence[0][2])
            wikiSentences = getWikiSentences(sentence[0][2])
            if (len(wikiSentences) > 1):
                ev_sentence = wikiSentences[int(sentence[0][3])]
                ev_sentence = ev_sentence['line'].replace('; ', ', ')
                ev_split = ev_sentence.split(', ')

                primary_component = ''
                prim_component_score = 0
                prim_component_label = 'neutral'

                for component in ev_split:

                    hypothesis = obj['claim']
                    premise = component

                    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                 max_length=max_length,
                                                 return_token_type_ids=True, truncation=True)

                    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
                    # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
                    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
                    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

                    outputs = model(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=None)

                    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

                    for i in range(0, len(predicted_probability)):
                         if (predicted_probability[i] > prim_component_score):
                             if i == 0:
                                 prim_component_label = 'SUPPORTS'
                                 primary_component = component
                                 prim_component_score = predicted_probability[i]
                             else:
                                 if (i == 2):
                                     prim_component_label = 'REFUTES'
                                     primary_component = component
                                     prim_component_score = predicted_probability[i]
                                 else:

                                     if (predicted_probability[i] > 0.95):
                                         prim_component_label = 'neutral'
                                         primary_component = component
                                         prim_component_score = predicted_probability[i]

                print("Premise:", ev_sentence)
                print("Hypothesis:", hypothesis)
                print("Label:", prim_component_label)
                print("Score:", prim_component_score)
