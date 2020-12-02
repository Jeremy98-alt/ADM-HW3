from bs4 import BeautifulSoup
import lxml
import requests

import os

import re

import csv

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from langdetect import detect

import math

import heapq

#RQ1.1
def get_urls(initial_url,n_pages):
    """a function to get the urls of each book

    Args:
        initial_url (str): the initial url we are working with
        n_pages (int): the number of pages we will be getting the books from
    """
    UrlsFiles = open("urlpages.txt", "w")

    for i in range(1, n_pages+1):
        url = initial_url +str(i)

        page = requests.get(url)
        soup = BeautifulSoup(page.content, features='lxml')
        for a in soup.find_all('a', class_="bookTitle"):
            UrlsFiles.write(a.get('href')+'\n')

    UrlsFiles.close()

#RQ2.2
def crawl_urls(n_pages, url_file):
    """a function to crawl the books, once the urls are given

    Args:
        n_pages (int): number of pages the urls are in
        url_file (str): name of the file the urls are in

    Returns:
        [type]: [description]
    """
    headpart = "https://www.goodreads.com"
    direct = "htmlpages" #Directory  
    parent_dir = "./" #Parent Directory path 
    pathAncestor = os.path.join(parent_dir, direct) #Path
    os.mkdir(pathAncestor) #create the folder in the path
    
    for i in range(1,n_pages+1):
        os.makedirs(os.path.join(pathAncestor, 'page ' + str(i))) # create sequentially the folders interested
    
    UrlsFiles = open(url_file, "r") # open the file in "read (r)" mode

    counter_pages = 0
    counter_html = 0
    for x in UrlsFiles: # crawl each Urls associated to the book to be sure to download the corresponding html article
        if counter_html % 100 == 0: # check every hundred html page to change the folder where we insert the html article
            counter_pages = counter_pages+1 

        counter_html = counter_html + 1

        subdirectory = pathAncestor + "/page " + str(counter_pages) # select the corresponding folder to insert the html article
        article_name = "/article_"+str(counter_html)+".html" # set the number of i-th book

        complete_path = subdirectory + article_name
        with open(complete_path, "wb") as ip_file:
            link = headpart + x
            try:
                page = requests.get(link)
            except:
                with open("failureRequest.txt", "a") as err_file: # if we loose the request book, we put into a file the link doesn't download well, then we set the "urlpages.txt" with these link
                    err_file.write(link)
                    err_file.close()

            soup = BeautifulSoup(page.text, features='lxml')

            ip_file.write(soup.encode('utf-8'))
            ip_file.close()

    UrlsFiles.close()
    
    return print("done")

#RQ1.3
def scrap_book(tsv_writer, article): 
    """the scraping function

    Args:
        tsv_writer ([type]): [description]
        article (str): name of the file we will be scraping from
    """
    global bookTitle, bookSeries, bookAuthors, ratingValue, ratingCount, reviewCount, Plot, NumberofPages, Published, Characters, Setting, Url; # set global variables to be sure that we consider into a scope this variables!
    
    with open(article, 'r', encoding="utf-8") as out_file: # for each html article downloaded scrape it!
        contents = out_file.read()
        soup = BeautifulSoup(contents, features="lxml") #parse the text
        
        # there are different excepts to be sure that if into the html article there isn't a information set it to empty string according to the professor requests
        
        #extract rating and review count
        try:
            ratings = soup.find_all('a', href="#other_reviews") #search the ratings in its
            rating_count = -1
            rating = -1
            for raiting in ratings:
                if raiting.find_all('meta', itemprop="ratingCount"):
                    ratingCount = raiting.text.replace('\n', '').strip().split(' ')[0].replace(',', '')
                elif raiting.find_all('meta', itemprop="reviewCount"):
                    reviewCount = raiting.text.replace('\n', '').strip().split(' ')[0].replace(',', '')
        except:
            ratingCount = " "
            reviewCount = " "
            
        #extract the book title
        try:
            bookTitle = soup.find_all('h1')[0].contents[0].replace('\n', '').strip()
        except:
            bookTitle = " "

        #extract the book authors
        try:
            bookAuthors = soup.find_all('span', itemprop='name')[0].contents[0]
        except:
            bookAuthors = " "

        #extract the book authors, we shoul FIX it.
        try:
            Plot = soup.find_all('div', id="description")[0].contents[3].text
            if detect(Plot) != "en":
                Plot = " "
        except:
            try:
                Plot = soup.find_all('div', id="description")[0].contents[1].text
                if detect(Plot) != "en":
                    Plot = " "
            except:
                Plot = " "
                

        #extract the date
        try:
            date = soup.find_all('div', id="details")[0].contents[3].text.replace('\n', '').strip().split()
            Published = date[1]+" "+date[2]+" "+date[3]
        except:
            Published = " "

        #Rating Value
        try:
            ratingValue = soup.find('span', itemprop="ratingValue").text.strip()
        except:
            ratingValue = " "

        #Number of pages
        try:
            NumberofPages = soup.find('span', itemprop="numberOfPages").text.split()[0]
        except:
            NumberofPages = " "

        #Title series
        try:
            bookSeries = soup.find_all('a', href= re.compile(r'/series/*'))[0].contents[0].strip()
        except:
            bookSeries = " "
            
        #Places
        try:
            Setting = []
            for places in soup.find_all('a', href= re.compile(r'/places/*')):
                Setting.append(places.text)
            Setting = ", ".join(Setting) if len(Setting)>=1 else " "
        except:
            Setting = " "

        #list of characters
        try:
            Characters = []
            for character in soup.find_all('a', href= re.compile(r'/characters/*') ):
                Characters.append(character.text)
            Characters = ", ".join(Characters) if len(Characters)>=1 else " "
        except:
            Characters = " "

        #extract the Url
        try:
            Url = soup.find_all('link')[0]["href"]
        except:
            Url = " "

        tsv_writer.writerow([bookTitle, bookSeries, bookAuthors, ratingValue, ratingCount, reviewCount, Plot, NumberofPages, Published, Characters, Setting, Url]) # insert the line into our article_i.tsv!

def get_scraping(input_path):
    """a function to apply the scraping to the input

    Args:
        input_path (str): the input from which to scrape
    """
    path = str("./"+input_path)

    filenames = os.listdir(path)
    for i in range(1, 301):
        filenames = os.listdir(path + '/' + str(i))

        for file in filenames:
            with open(path + '/' + str(i) + './article_'+str(file.split("_")[1].replace(".html", ""))+'.tsv', 'w', encoding="utf-8", newline='') as out_file: # create for each html article its article_i.tsv according to the professor requests!
                tsv_writer = csv.writer(out_file, delimiter='\t')
                tsv_writer.writerow(['bookTitle', 'bookSeries', 'bookAuthors', 'ratingValue', 
                                'ratingCount', 'reviewCount', 'Plot', 'NumberofPages', 'Published',
                                'Characters', 'Setting', 'Url'])
                scrap_book(tsv_writer, path + '/' + str(i) + "/" + file)

#creating the final dataset
def finaldataset_tsv(initial_path):
    """a function that creates the tsv file of the specified input

    Args:
        initial_path (str): where to get the files from
        distinct_range (int): starting point from which to get the files within the given path
    """
    path = str("./"+initial_path)
    suffix = ".tsv"
    filenames = os.listdir(path)
    data2 = pd.DataFrame()
    for i in range(1, 301):
        filenames = os.listdir(path + '/' + str(i))
        for file in filenames:
            if file.endswith(suffix): # check the .tsv suffix because there are .html extension
                with open(path + '/' + str(i) + '/article_'+str(file.split("_")[1]), 'r', encoding="utf-8", newline='') as out_file:
                        df = pd.read_csv(out_file,sep = "\t")
                        if  df.loc[1,"Plot"] != " " and df.loc[1,"bookTitle"] != " ": # check if the  article_i.tsv contains the main information, if it is insert in the final dataset
                            data2 = pd.concat([data2,df])
                            
    with open("finaloutput.tsv", "w", encoding="utf-8", newline="") as text_file: text_file.write(data2.to_csv(index=False))

#RQ2
def openMainDataset():
    df = pd.read_csv("../finaloutput.csv", sep=",")
    return df
    
def cleaningDataset(df):
    """a function to clean the dataset

    Args:
        df (dataframe): the dataset to clean

    Returns:
        dataframe: the clean dataset
    """
    for i, row in df.iterrows():
        tokenizer = RegexpTokenizer("[\w']+")  # import the tokenizer punctuation
        
        df.at[i, 'Plot'] = tokenizer.tokenize(df.at[i, 'Plot'].lower()) # remove the punctuation
        df.at[i, 'Plot'] = [w for w in df.at[i, 'Plot'] if not w in set(stopwords.words('english'))]  # words in english to avoid few data for the cleaning data
        df.at[i, 'Plot'] = [PorterStemmer().stem(word) for word in df.at[i, 'Plot']] # contesto
    return df

def createVocabulary(df):
    """a function to create a vocabulary, given a dataset

    Args:
        df (dataframe): the dataset from which we will be creating our vocabulary

    Returns:
        dictionary: the vocabulary we needed
    """
    vocabulary = {}
    for i, row in df.iterrows():
        for word in df.at[i, 'Plot']:
            if word in vocabulary.keys():
                if "document_"+str(i) not in vocabulary[word]:
                    vocabulary[word].append("document_"+str(i))
            else:
                vocabulary[word] = ["document_"+str(i)]
    return vocabulary

def create_csv(vocabulary):
    """create a csv file containing the vocabulary and how often each word is repeated in any given book

    Args:
        name_file (str): the name of the csv file we will be creating
    """
    with open('./vocabulary.tsv', 'w', encoding="utf-8", newline='') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(["Word", "Term_id", "Document_List"])

        list_of_words = vocabulary.keys()
        for id, word in enumerate(list_of_words, 1):
            term_id_i = "term_id_"+str(id)
            tsv_writer.writerow([word, term_id_i, vocabulary[word]]) 

#RQ2.1
def cleanQuery(row):
    """a function that cleans the input query, so that it can be used in the subsequent analysis

    Args:
        row (str): the query we will be working on

    Returns:
        str: the clean query
    """
    row = row.split(" ") # split the string query
    for element in row:
        tokenizer = RegexpTokenizer("[\w']+")  # import the tokenizer punctuation
        element = tokenizer.tokenize(element.lower()) # remove the punctuation
    
    row = [w for w in row if not w in set(stopwords.words('english'))] # words in english to avoid few data for the cleaning data
    row = [PorterStemmer().stem(word) for word in row]
    
    return row

#convert it into a list, because the dataset mark the information like text and not like list data structure
def make_it_list(word):
    """??? not sure what the types are

    Args:
        word ([type]): [description]

    Returns:
        [type]: [description]
    """
    for sym in ["[", "]", "'"]:
        word = word.replace(sym, "")
    word = word.split(", ")
    
    return word

#open vocabulary
def open_vocabulary():
    """a quick function to open the dataset vocabulary

    Returns:
        dataframe: the vocabulary dataset
    """
    vocabulary = pd.read_csv('vocabulary.tsv',sep="\t")
    vocabulary = vocabulary.drop_duplicates(subset=['Word'])
    vocabulary = vocabulary.reset_index(drop=True)

    return vocabulary

def inverted_list(vocabulary):
    #create the inverted list
    inv_lst = {}
    for i, row in vocabulary.iterrows():
        term = vocabulary.at[i, "Term_id"]
        inv_lst[term] = make_it_list(vocabulary.at[i, "Document_List"])
    
    #and save it as csv file like the previous consideration about the vocabulary .csv file!
    with open('inv_lst.csv', 'w') as f:
        for key in inv_lst.keys():
            f.write("%s:%s\n"%(key,inv_lst[key]))

# like the function title
def open_and_convert_inv_lst(inv_lst_csv):
    reader = csv.reader(open(inv_lst_csv, 'r'), delimiter=":") # recall a function to open the inverted list..
    
    inv_lst = {}
    for row in reader:
        k, v = row
        inv_lst[k] = v
        
    for key in inv_lst:
        inv_lst[key] = make_it_list(inv_lst[key]) # convert and make it dictionary
    
    return inv_lst

# map the interested word with corresponding term_id_i
def map_terms_id(vocabulary, cleanQString):
    # find each term_id
    term_id = []           # this is another function useful for mapping the term_id_i with the word into the vocabulary...
    for token in cleanQString:
        term = vocabulary.loc[vocabulary["Word"] == token, "Term_id"].values[0]
        term_id.append(term)
        
    return term_id

def search_engine1(cleanQString, inv_lst_csv, vocabulary, df_copy, df):
    """the first search engine we have created, which only looks at whether the words of the query are found in the plots of the books

    Args:
        cleanQString (str): the clean query
        inv_lst_csv (str): the name of the csv file containing the csv file
    """
    inv_lst = open_and_convert_inv_lst(inv_lst_csv)
    
    term_id = map_terms_id(vocabulary, cleanQString)
    
    # find the common documentsdd
    intersection_list = []
    for term in term_id:
        if not intersection_list:
            intersection_list = inv_lst[term]
        else:
            intersection_list = set(intersection_list).intersection(set(inv_lst[term]))
    
    new_df = pd.DataFrame(columns=['bookTitle', 'Plot', 'Url'])
    for row in intersection_list:
        i = int(row.split("_")[1])
        #append row to the dataframe
        new_row = {'bookTitle': df_copy.loc[i, "bookTitle"], 'Plot': df.loc[i, "Plot"], 'Url': df_copy.loc[i, "Url"]}
        new_df = new_df.append(new_row, ignore_index=True)
    
    return new_df

#RQ2.2
def new_inv_lst(inv_lst_csv, vocabulary, df):
    """create a new inverted list starting from the one we have already defined (why?)

    Args:
        inv_lst_csv (str): the inverted list we start from
        vocabulary (DATAFRAME???): the vocabulary we work with to create the inverted list
    """
    inv_lst = open_and_convert_inv_lst(inv_lst_csv)
        
    #create a second inverted list
    inv_lst2 = {}
    
    for i, row in vocabulary.iterrows():
        lst_doc = make_it_list(vocabulary.at[i, 'Document_List'])
        
        result = []
        for doc in lst_doc:
            number_doc = int(doc.split("_")[1])
            
            interested_row = df.at[number_doc, "Plot"]
            
            interested_word = vocabulary.at[i, "Word"] #i-th word
            
            tf = interested_row.count(interested_word) / len(interested_row)
            
            idf = math.log(len(df)/len(lst_doc))
            
            tf_idf = round(tf * idf, 3)
            
            result.append((doc, tf_idf))
            
        inv_lst2[vocabulary.at[i, "Term_id"]] = result
        
    return inv_lst2

#some functions we will need for the second search engine
def make_it_list2Version(word):
    """cleans up the input by removing unnecessary brackets

    Args:
        word (string): the string that need cleaning up

    Returns:
        list: a list containing the character???
    """
    for sym in ["[", "]", "'", "(", ")"]:
        word = word.replace(sym, "")
    word = word.split(", ")
    
    return list(word)

# map each term with the corresponding word
def returnTermId(token, vocabulary):
    """a function to find each term id

    Args:
        token (str): the words which ids we need

    Returns:
        [type]: [description]????? df
    """
    return vocabulary.loc[vocabulary["Word"] == token, "Term_id"].values[0]

# create the documents list to make the cosine similarity logic
def create_documents_list(df, inv_lst2, vocabulary):
    documents = []                                             
    for i, row in df.iterrows():
        tokens = {}
        for token in df.at[i, "Plot"]:
            tuple_list_values = inv_lst2[returnTermId(token, vocabulary)]

            for x in tuple_list_values:
                if int(x[0].split("_")[1]) == i:
                    tokens[token] = x[1]  
                    break

        documents.append(tokens)
    
    return documents

def similarity_score(df, inv_lst2, vocabulary, cleanQString):
    """a function used to find the similarity score of the books in the dataframe, given a certain inverted list

    Args:
        df ([type]): [description]
        inv_lst2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    documents = create_documents_list(df, inv_lst2, vocabulary)

    # HERE STARTS THE SIMILARITY SCORE!
    top_k_documents = []
    for i, row in df.iterrows():
        card_d_i = 1 / math.sqrt( sum(documents[i].values()) )

        somma = 0
        for token in cleanQString:
            try:
                somma += documents[i][token]
            except:
                somma += 0

        cosine_similarity = card_d_i * somma

        top_k_documents.append([round(cosine_similarity, 2), "document_"+str(i)])
        
    return top_k_documents

# convert the list of top k documents in heap structure
def heap_k_documents(top_k_documents):
    heapq.heapify(top_k_documents) 
    show_top_k_documents = (heapq.nlargest(5, top_k_documents)) 
    return show_top_k_documents

def search_engine2(df_copy, df, inv_lst2, vocabulary, cleanQString):
    """the second search engine

    Args:
        new_df ([type]): [description]

    Returns:
        [type]: [description]
    """
    top_k_documents = similarity_score(df_copy, inv_lst2, vocabulary, cleanQString)
    
    # make it heap!
    show_top_k_documents = heap_k_documents(top_k_documents)
    
    new_df = pd.DataFrame(columns=['bookTitle', 'Plot', 'Url', 'Similarity'])
    for row in show_top_k_documents:
        i = int(row[1].split("_")[1])

        #append row to the dataframe
        new_row = {'bookTitle': df_copy.loc[i, "bookTitle"], 'Plot': df.loc[i, "Plot"], 'Url': df_copy.loc[i, "Url"], 'Similarity': row[0]}
        new_df = new_df.append(new_row, ignore_index=True)
    return new_df