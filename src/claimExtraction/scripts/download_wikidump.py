import argparse
import requests
import re
import bz2
import os
from bs4 import BeautifulSoup
from simple_term_menu import TerminalMenu


def getDumpUrls():
        
    response = requests.get("https://dumps.wikimedia.org/enwiki/latest/")
      
    print(response.status_code)
    
    x = response.text.replace('</a> ', '</a>  ')
            
    lines = x.split("\n")
    
    final = []
    
    for i in range(4, len(lines)-3):
        temp = lines[i].split("  ")
        for j in range(0, len(temp)):
            if (temp[j] != ""):
                final.append(temp[j].strip())          
    
    objs = []
                
    for i in range(0, len(final)-3, 3):
        
        a_element = final[i]
        date = final[i+1]
        size = str(round(int(final[i+2]) * 10e-6, 2))
        
        soup = BeautifulSoup(a_element, 'html.parser')
    
        url = soup.find("a").get('href')
                
        if re.search('enwiki-latest-pages-articles-multistream[0-9]', url) and (float(size) > 1000):
            url_info = url.replace('enwiki-latest-pages-articles-multistream', '')
            url_info = url_info.replace('.xml-p', ' ')
            url_info = url_info.replace('p', ' ')
            url_info = url_info.replace('.bz2', '')
            index, page_start, page_end = url_info.split(' ')[0], url_info.split(' ')[1], url_info.split(' ')[2]
                        
            obj = {'index': index, 'page_start': page_start, 'page_end': page_end, 'date': date, 'size': size, 'url': url}
            objs.append(obj)
            
    for obj in objs:
        if (os.path.isfile('./../data/wikidumps/' + obj['url'].replace('.bz2', ''))):
            objs.remove(obj)
    
    objs = sorted(objs, key=lambda k: int(k['page_start']))
                
    return objs

def downloadDumps(chosen_dumps):
    
    i = 0
    
    for chosen_dump in chosen_dumps:
        
        i = i + 1
        print("Downloading " + str(chosen_dump['url']) + " (" + str(i) + "/" + str(len(chosen_dumps) * 2) + ")")
        url = "https://dumps.wikimedia.org/enwiki/latest/" + chosen_dump['url']
        r = requests.get(url, allow_redirects=True)
        open('./../data/wikidumps/' + chosen_dump['url'], 'wb').write(r.content)
        i = i + 1
        print("Unzipping " + str(chosen_dump['url']) + " (" + str(i) + "/" + str(len(chosen_dumps)* 2) + ")")
        
        with bz2.open('./../data/wikidumps/' + chosen_dump['url'], "rb") as f:
            content = f.read()
        
        open('./../data/wikidumps/' + chosen_dump['url'].replace('.bz2', ''), 'wb').write(content)
        
        os.remove('./../data/wikidumps/' + chosen_dump['url'])
        
parser = argparse.ArgumentParser()
parser.add_argument('--all', required=False, action='store_true')

args = parser.parse_args()
        
dumpUrls = getDumpUrls()

if (dumpUrls != []):
    
    if (args.all == False):
        menu_entries = []

        for dumpUrl in dumpUrls:
            menu_entries.append((dumpUrl['index'] + ' '*(3 - len(dumpUrl['index']))) + "    " + dumpUrl['date'] + "    " + (dumpUrl['size'] + ' '*(8 - len(dumpUrl['size']))) + "    " + (dumpUrl['page_start'] + ' '*(8 - len(dumpUrl['page_start']))) + "    " + dumpUrl['page_end'])
                
        terminal_menu = TerminalMenu(menu_entries, title='Index no.    Date                 Size (mb)   Start       End', multi_select=True)
        choice_indices = terminal_menu.show()
        chosen_dumps = [dumpUrls[index] for index in choice_indices]
    else:
        chosen_dumps = [dumpUrls[index] for index in range(0, len(dumpUrls)-1)]
    
    downloadDumps(chosen_dumps)

else:
    print("All wikidumps downloaded.")

