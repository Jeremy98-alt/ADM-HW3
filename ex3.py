import pandas as pd
import json
import math
import heapq
import math

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from nltk.tokenize import RegexpTokenizer

################################ RQ3 #####################################

def openCSV(file):
    return pd.read_csv("../finalOutput.csv", sep=",") # open the final dataset

# clean the text, create the bag of words
def clean_text(text):
    stop_words = set(stopwords.words('english')) # obtain the stop words
    good_words = [] # save the correct words to consider like tokens
    tokenizer = RegexpTokenizer("[\w']+") # function to recognize the tokens
    words = tokenizer.tokenize(text) # tokenize the text 
    for word in words:
        # check if the word is lower and it isn't a stop word or a number
        if word.lower() not in stop_words and word.isalpha(): 
            word = PorterStemmer().stem(word) # use the stemmer function
            good_words.append(word.lower()) # insert the good token to lower case
        
    return good_words

# cleaning the text column values, apply the function for each value through the corresponding column
def cleaning_value_columns(df1):
    df1['bookTitle'] = df1.bookTitle.apply(lambda x: clean_text(x))
    df1['bookAuthors'] = df1.bookAuthors.apply(lambda x: clean_text(x))
    df1['Plot'] = df1.Plot.apply(lambda x: clean_text(x))
    return df1
    
# create the new vocabulary
def create_newVocabulary(df1):
    vocabulary2 = {}
    for i, row in df1.iterrows():
        for column in ["bookTitle", "bookAuthors", "Plot"]: # insert the tokens into the new vocabulary
            if len(df1.at[i, column]) > 0:  # check if the list is empty or not to avoid the eventually error
                for word in df1.at[i, column]: # bring the token from the list
                    if word in vocabulary2.keys(): # insert the token into the vocabulary with the documents where this is present
                        if i not in vocabulary2[word]:
                            vocabulary2[word].append(i)
                    else:
                        vocabulary2[word] = [i]
    return vocabulary2

# create the inverted list according to the professors' requests
def create_newInvertedList(vocabulary2):
    inv_lst = {}
    indexes = list(vocabulary2.keys()) # return the indexes list of the vocabulary
    for key in vocabulary2.keys():
        term_id = indexes.index(key) # find the corresponding id from the vocabulary
        inv_lst[term_id] = vocabulary2[key] # insert the list of documents into the inverted list
    
    return inv_lst

# map the interested word with corresponding term_id_i
def map_terms_id(vocabulary2, cleanQString):
    # find each term_id
    term_id = []  # this is another function useful for mapping the term_id_i with the word into the vocabulary
    indexes = list(vocabulary2.keys()) # return the indexes list of the vocabulary
    for token in cleanQString:
        term = indexes.index(token)
        term_id.append(term) # append the id that we want to make the score
        
    return term_id

# clean user's query
def cleanQuery(query):
    cleanQString = query.split(" ")
    
    stop_words = set(stopwords.words('english')) # obtain the stop words
    good_words = [] # save the correct words to consider like tokens
    tokenizer = RegexpTokenizer("[\w']+") # function to recognize the tokens
    
    for word in cleanQString:
        word = tokenizer.tokenize(word) # tokenize the text
        for w in word:
            if w.lower() not in stop_words and w.isalpha(): 
                w = PorterStemmer().stem(w) # use the stemmer function
                good_words.append(w.lower()) # insert the good token to lower case
    
    return good_words

# the new search engine 1
def newsearch_engine1(cleanQString, vocabulary2, df1, df, inv_lst):
    term_id = map_terms_id(vocabulary2, cleanQString) # return the corresponding id of those terms

    # find the common documents where those terms are present
    intersection_list = []
    for term in term_id:
        if not intersection_list:
            intersection_list = inv_lst[term] # if the intersection list is empty insert the first list of the first token
        else:
            intersection_list = set(intersection_list).intersection(set(inv_lst[term])) # make the intersection, this respect the properties of the sets

    new_df = pd.DataFrame(columns=['bookTitle', 'Plot', 'Url']) # create the new dataset according to the professors' requests
    for row in intersection_list:
        #append row to the dataframe
        new_row = {'bookTitle': df.loc[row, "bookTitle"], 'Plot': df.loc[row, "Plot"], 'Url': df.loc[row, "Url"]}
        new_df = new_df.append(new_row, ignore_index=True)
        
    return new_df

################# RQ3.2

# define the new inv_lst2 according to create the new score
def new_inv_lst2(vocabulary2, df1):
    inv_lst2 = {}

    indexes = list(vocabulary2.keys())
    for key in vocabulary2.keys():
        lst_doc = vocabulary2[key]

        result = []
        for doc in lst_doc:
            tf_idf = []
            for column in ["bookTitle", "bookAuthors", "Plot"]: # insert all tokens present in those columns
                interested_row = df1.at[doc, column] # extract the list of tokens from a proper column

                interested_word = key #i-th word

                # insert this construct because the interested_row could be empty, so put 0 like the term frequency
                try:
                    tf = interested_row.count(interested_word) / len(interested_row) 
                except:
                    tf = 0

                idf = math.log(len(df1)/len(lst_doc))

                tf_idf.append(round(tf * idf, 3))

            result.append((doc, round(sum(tf_idf)/3, 3))) # normalize the result

        inv_lst2[indexes.index(key)] = result # insert the result into the inverted list
        
    return inv_lst2

# define the documents of tokens with tf-idf for each token corresponding to the i-th document
def documents_list(vocabulary2, inv_lst2, df1):
    documents = [] 
    indexes = list(vocabulary2.keys()) # return the indexes list of the vocabulary
    for i, row in df1.iterrows():
        tokens = {} # insert the tokens and put its tf_idf score mapped in the i-th document
        for col in ["bookTitle", "bookAuthors", "Plot"]: # check those three columns
            for token in df1.at[i, col]:
                    tuple_list_values = inv_lst2[indexes.index(token)] # consider the list of documents

                    for x in tuple_list_values: # catch the documents where there is the tf_idf score of the token present into the document
                        if x[0] == i:
                            tokens[token] = x[1] # consider the score
                            break # break we find the interested term_id of the documents, we catch the score

        documents.append(tokens) # append the tokens with their tf_idf
        
    return documents

# new score and return top_k_documents according to the similarity formula and the boost considered by us
def newscore(df1, documents, cleanQString):
    top_k_documents = []
    for i, row in df1.iterrows():
        card_d_i = 1 / math.sqrt( sum(documents[i].values()) )

        somma = 0
        for token in cleanQString:
            try: # if the token isn't present the sum is equal to 0
                somma += documents[i][token]
            except:
                somma += 0

        cosine_similarity = card_d_i * somma

        top_k_documents.append([round(cosine_similarity, 2), i])  
     
    # boost it!
    for boost in top_k_documents:
        if float(df1.at[boost[1], "ratingValue"]) > 3.5 and int(df1.at[boost[1], "ratingCount"]) > 3500000:
            boost[0] += 2.0
        
    return top_k_documents

# sort and show the new_df according to the new score found
def newsearch_engine2(vocabulary2, df, df1, cleanQString, k):
    # create the new inverted list
    inv_lst2 = new_inv_lst2(vocabulary2, df1)
    
    # create the new document list
    documents = documents_list(vocabulary2, inv_lst2, df1)
    
    # obtain the new score so.. the top_k documents!
    top_k_documents = newscore(df1, documents, cleanQString)
    
    heapq.heapify(top_k_documents)
    show_top_k_documents = (heapq.nlargest(k, top_k_documents)) 
    
    new_df = pd.DataFrame(columns=['bookTitle', 'Plot', 'Url', 'New-Score'])
    for row in show_top_k_documents:  #append row to the dataframe
        new_row = {'bookTitle': df.loc[row[1], "bookTitle"], 'Plot': df.loc[row[1], "Plot"], 'Url': df.loc[row[1], "Url"], 'New-Score': row[0]}
        new_df = new_df.append(new_row, ignore_index=True)
    
    return new_df







