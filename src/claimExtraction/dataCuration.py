import re
import requests
import json
import random
import time
from newspaper import Article
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

punkt_param = PunktParameters()
punkt_param.abbrev_types = set(open("data/ignore_delimiters.txt", "r").read().split())
sentence_splitter = PunktSentenceTokenizer(punkt_param)

ignoreTokens = open("data/stopwords.txt", "r").read().split()
ignoreChars = open("data/ignore_chars.txt", "r").read().split()
#ignoreChars = [".", ",", ":", ";", "\'", "''", "`", "?", "%", "!", "$","0","1","2","3","4","5","6","7","8","9"]
ignores = ignoreChars+ignoreTokens

def getArticleUrls():
    
    articleUrls = []
    
    topics = open("data/topics.txt", "r").read().split()
    
    #publishStartDate = "2021-06-01"
    #publishEndDate = "2021-06-10"
    #country = "us"
    language = "en"
    #pageSize = "20"
    
    for topic in topics:
        for page in range(1, 6):
            response = requests.get("https://newsapi.org/v2/everything?q=" + topic + "&language=" + language + "&page=" + str(page) + "&sortBy=relevancy&apiKey=383e6061e4414bf0b1211f1639f1e433")
            print(response.status_code)
            
            apiDump = json.loads(json.dumps(response.json()))                
            articles = apiDump['articles']
            
            for art in articles:
                articleUrls.append(art['url'])
                
        time.sleep(60)
    
    return list(set(articleUrls))

articleUrls = getArticleUrls()

finalSentences = []

articleCount = 0

for url in articleUrls:
    
    articleCount = articleCount + 1
    
    print(str(articleCount) + "/" + str(len(articleUrls)))
    
    article = Article(url)
    
    try:
        article.download()
        article.parse()
        
        article.text = re.sub('\n\n', ' ', article.text).strip()
        split_sent = sentence_splitter.tokenize(article.text)
            
        hyperlinkNodes = article.clean_top_node.xpath('//p//a') + article.clean_top_node.xpath('//li//a')
        parentNodes = article.clean_top_node.xpath('//a/..')
        
        hyperlinkNodes = [hlNode for hlNode in hyperlinkNodes if ((hlNode.text != None) and (hlNode.tail != None))]
        parentNodes = [pNode for pNode in parentNodes if (pNode.text != None)]
    except:
        print("download failed")
    
    
    for pNode in parentNodes:
        for sent in split_sent:
            if (sent in pNode.text):
                pNode.text = pNode.text.replace(sent, '')
    
    sentences = []
    
    for parentNode in parentNodes:
        tail = ""   
        for node in hyperlinkNodes:             
            if node in parentNode:
                tail = tail + node.text + node.tail            
                if (node.tail[len(node) - 1] == '.'):
                    sentences.append(parentNode.text + tail)
                    tail = ""
        
    claimSentences = []                
    
    for hyperlinkNode in hyperlinkNodes:
        for sentence in sentences:
            sentence = re.sub('\xa0', ' ', sentence).strip()
            split = sentence_splitter.tokenize(sentence)
            for i in split:
                if (hyperlinkNode.text in i) and (i not in claimSentences):
                    claimSentences.append(i)
    
    for sent in split_sent:
        if (sent in claimSentences):
            finalSentences.append('[' + sent + ']')
        else:
            finalSentences.append(sent)
            
print(finalSentences)

extractedSentences = []

claims = 0
not_claims = 0

for finalSentence in finalSentences:
    
    if (len(finalSentence.split()) >= 4):
        if ((finalSentence[0] == '[') and (finalSentence[len(finalSentence)-1] == ']')):
            finalSentence = finalSentence[1:len(finalSentence)-1]
            label = 'CLAIM'
            claims = claims + 1
        else:
            label = 'NOTCLAIM'
            not_claims = not_claims + 1
        
        tagged_sent = nltk.pos_tag(finalSentence.split(), tagset='universal')
        
        words, tags = zip(*tagged_sent)
          
        tokens = []
        
        for token in nltk.word_tokenize(finalSentence):
            if (token not in ignores):
                tokens.append(token)
        
        clean_sentence = finalSentence.lower()
        clean_sentence = re.sub(r"\'t", " not", clean_sentence)
        clean_sentence = re.sub(r'(@.*?)[\s]', ' ', clean_sentence)
        clean_sentence = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', clean_sentence)
        clean_sentence = re.sub(r'[^\w\s\?]', ' ', clean_sentence)
        clean_sentence = re.sub(r'([\;\:\|•«\n])', ' ', clean_sentence)
        clean_sentence = " ".join([word for word in clean_sentence.split()
          if word not in ignores
          or word in ['not', 'can']])

        clean_sentence = re.sub(r'\s+', ' ', clean_sentence).strip()
        
        if (clean_sentence != None):                
            obj = { "label": label, "sentence_text": clean_sentence, "tokens": tokens, "pos": tags, "sentence": finalSentence }
                
        extractedSentences.append(json.dumps(obj))

print("No. claims: " + str(claims))
print("No. non-claims: " + str(not_claims))

extractedSentences = list(set(extractedSentences))

claim = 0
not_claim = 0

classBalanced = []

for sentence in extractedSentences:
    if (json.loads(sentence)['label'] == 'CLAIM'):
        if (claim <= not_claim):
            classBalanced.append(sentence)    
            claim = claim + 1
    else:
        if (claim >= not_claim):
            classBalanced.append(sentence)    
            not_claim = not_claim + 1
            
extractedSentences = classBalanced

random.shuffle(extractedSentences)
        
train_file = open("output/train.json", "w")
for line in extractedSentences[0:int(len(extractedSentences)*0.6)]:
    train_file.write(line + '\n')

train_file.close()

train_file = open("output/train.label", "w")
for line in extractedSentences[0:int(len(extractedSentences)*0.6)]:
    train_file.write(json.loads(line)['label'] + ' ' + json.loads(line)['sentence_text'] + '\n')

train_file.close()

dev_file = open("output/dev.json", "w")
for line in extractedSentences[int(len(extractedSentences)*0.6):int(len(extractedSentences)*0.8)]:
    dev_file.write(line + '\n')

dev_file.close()

dev_file = open("output/dev.label", "w")
for line in extractedSentences[int(len(extractedSentences)*0.6):int(len(extractedSentences)*0.8)]:
    dev_file.write(json.loads(line)['label'] + ' ' + json.loads(line)['sentence_text'] + '\n')

dev_file.close()

test_file = open("output/test.json", "w")
for line in extractedSentences[int(len(extractedSentences)*0.8):len(extractedSentences)]:
    test_file.write(line + '\n')

test_file.close()

test_file = open("output/test.label", "w")
for line in extractedSentences[int(len(extractedSentences)*0.8):len(extractedSentences)]:
    test_file.write(json.loads(line)['label'] + ' ' + json.loads(line)['sentence_text'] + '\n')

test_file.close()