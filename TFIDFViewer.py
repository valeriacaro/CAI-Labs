"""
.. module:: TFIDFViewer

TFIDFViewer
******

:Description: TFIDFViewer

    Receives two paths of files to compare (the paths have to be the ones used when indexing the files)

:Authors:
    bejar

:Version:

:Date:  05/07/2017
"""

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.client import CatClient
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q

import argparse

import numpy as np

__author__ = 'bejar'

def search_file_by_path(client, index, path):
    """
    Search for a file using its path

    :param path:
    :return:
    """
    s = Search(using=client, index=index)
    q = Q('match', path=path)  # exact search in the path field
    s = s.query(q)
    result = s.execute()

    lfiles = [r for r in result]
    if len(lfiles) == 0:
        raise NameError(f'File [{path}] not found')
    else:
        return lfiles[0].meta.id


def document_term_vector(client, index, id):
    """
    Returns the term vector of a document and its statistics a two sorted list of pairs (word, count)
    The first one is the frequency of the term in the document, the second one is the number of documents
    that contain the term

    :param client:
    :param index:
    :param id:
    :return:
    """
    termvector = client.termvectors(index=index, id=id, fields=['text'],
                                    positions=False, term_statistics=True)

    file_td = {}
    file_df = {}

    if 'text' in termvector['term_vectors']:
        for t in termvector['term_vectors']['text']['terms']:
            file_td[t] = termvector['term_vectors']['text']['terms'][t]['term_freq']
            file_df[t] = termvector['term_vectors']['text']['terms'][t]['doc_freq']
    return sorted(file_td.items()), sorted(file_df.items())


def toTFIDF(client, index, file_id):
    """
    Returns the term weights of a document

    :param file:
    :return:
    """

    # Get the frequency of the term in the document, and the number of documents
    # that contain the term
    file_tv, file_df = document_term_vector(client, index, file_id)
    
    #file_tv are the term frequencies per document sorted
    #file_df are the number of documents that contain a term

    # Get the max frequency of all terms
    max_freq = max([f for _, f in file_tv])

    # Get the total number of documents
    dcount = doc_count(client, index)

    # Declare a list for the tfidf weights
    tfidfw = []
    
    for (t, w),(_, df) in zip(file_tv, file_df):
        
        # Compute the weight of the term
        tf = w/max_freq
        idf = np.log2(dcount/df)
        weight = tf*idf
        
        # Store the term and its weight
        tfidfw.append([t, weight])

    # Return the normalized weights
    return normalize(tfidfw)

def print_term_weigth_vector(twv):
    """
    Prints the term vector and the correspondig weights
    :param twv:
    :return:
    """
    
    # Print every pair (term, weight)
    for pair in twv:
        print(pair)
    return


def normalize(tw):
    """
    Normalizes the weights in t so that they form a unit-length vector
    It is assumed that not all weights are 0
    :param tw:
    :return:
    """

    # Compute the norm
    t_sum = 0
    for (t,w) in tw:
        t_sum += w*w
    norm = np.sqrt(t_sum)
    
    # Declare a list to store the normalized terms
    twnorm = []
    
    # For each term, append the term and its normalized weight
    for (t, w) in tw:
        twnorm.append((t, w/norm))
        
    # Return the normalized weights list
    return twnorm


def cosine_similarity(tw1, tw2):
    """
    Computes the cosine similarity between two weight vectors, terms are alphabetically ordered
    :param tw1:
    :param tw2:
    :return:
    """

    # Initialize the cosine similarity and two counters
    cs = 0
    i = 0
    j = 0
    
    # Travel through both lists
    while (i < len(tw1)) and (j < len(tw2)):
        
        # If both words are the same, compute its cosine similarity and add it to the total
        if tw1[i][0] == tw2[j][0]:
            cs += tw1[i][1]*tw2[j][1]
            
            # Go to the next position of both lists
            i+=1
            j+=1
            
        # If two words are different, go to the next position in the list which has a lower term (alphabetically speaking)
        elif tw1[i][0] < tw2[j][0]:
            i+=1
        else:
            j+=1
            
    # Return the total cosine similarity
    return cs

def doc_count(client, index):
    """
    Returns the number of documents in an index

    :param client:
    :param index:
    :return:
    """
    return int(CatClient(client).count(index=[index], format='json')[0]['count'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=None, required=True, help='Index to search')
    parser.add_argument('--files', default=None, required=True, nargs=2, help='Paths of the files to compare')
    parser.add_argument('--print', default=False, action='store_true', help='Print TFIDF vectors')

    args = parser.parse_args()


    index = args.index

    file1 = args.files[0]
    file2 = args.files[1]

    client = Elasticsearch(timeout=1000)

    try:

        # Get the files ids
        file1_id = search_file_by_path(client, index, file1)
        file2_id = search_file_by_path(client, index, file2)

        # Compute the TF-IDF vectors
        file1_tw = toTFIDF(client, index, file1_id)
        file2_tw = toTFIDF(client, index, file2_id)

        if args.print:
            print(f'TFIDF FILE {file1}')
            print_term_weigth_vector(file1_tw)
            print ('---------------------')
            print(f'TFIDF FILE {file2}')
            print_term_weigth_vector(file2_tw)
            print ('---------------------')

        print(f"Similarity = {cosine_similarity(file1_tw, file2_tw):3.5f}")

    except NotFoundError:
        print(f'Index {index} does not exists')
