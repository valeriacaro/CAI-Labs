from __future__ import print_function
from tkinter import W
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.client import CatClient
import numpy as np

import argparse

from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q

__author__ = 'bejar'

def search_file_by_path(s, path):
	"""
	Search for a file using its path

	:param path:
	:return:
	"""
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
	max_freq = max([f for _, f in file_tv])

	dcount = doc_count(client, index)
	tfidfw = []
	for (t, w),(_, df) in zip(file_tv, file_df):
		tf = w/max_freq
		idf = np.log2(dcount/df)
		weight = tf*idf
		tfidfw.append([t, weight])
	return normalize(tfidfw)

def print_term_weigth_vector(twv):
	"""
	Prints the term vector and the correspondig weights
	:param twv:
	:return:
	"""
	for p in twv:
		print('('+str(p[0])+','+str(p[1])+')')
	return None


def normalize(tw):
	"""
	Normalizes the weights in t so that they form a unit-length vector
	It is assumed that not all weights are 0
	:param tw:
	:return:
	"""
	norm = 0
	for (_,w) in tw:
		norm += w**2
	i = 0
	for (t,w) in tw:
		tw[i] = (t,w/np.sqrt(norm))
		i += 1
	return tw


def doc_count(client, index):
	"""
	Returns the number of documents in an index

	:param client:
	:param index:
	:return:
	"""
	return int(CatClient(client).count(index=[index], format='json')[0]['count'])

def get_docs_from_query(k, query, s):
	if query is not None:
		q = Q('query_string',query=query[0])
		for i in range(1, len(query)):
			q &= Q('query_string',query=query[i])
		s = s.query(q)
		result = s.execute()
		if (result.hits.total['value']>k):
			result = s[0:k].execute()
		return result

	
def get_summed_weight_words (k_doc_result, client, index):
	#agafem tots els documents amb tots els seus pessos a les paraules i els sumem (element que aniria dividit per k i multiplicat per beta)
	dictionary = {}
	for i in range(0, len(k_doc_result)):
		path = k_doc_result[i]["path"]
		file_id = search_file_by_path(s, path)
		v = toTFIDF(client, index, file_id)
		for elem in v:
			if elem in dictionary: #we update weight if found
				dictionary[elem[0]] = dictionary[elem[0]] + elem[1]
			else:
				dictionary[elem[0]]=elem[1] #we create a new entry if not found
	return dictionary

def convert_query_to_dict(query):
	dictionary = {}
	for elem in query:
		split_word = elem.split('^')
		if(len(split_word) > 1):
			w = float(split_word[1])
		else:
			w = 1.0
		par = split_word[0]
		if par in dictionary:
			dictionary[par] += w
		else:
			dictionary[par] = w
	return dictionary

def prune_dict (R,updated_query_dict):
	pruned_dict = {}
	i = 0
	for elem in updated_query_dict:
		if i < R:
			pruned_dict[elem] = updated_query_dict[elem]
			i = i + 1
		else:
			return pruned_dict

def compute_rocchio(alpha, beta, index, k, k_doc_result, query, client):
	query_dict = convert_query_to_dict(query)
	words_dict = get_summed_weight_words(k_doc_result, client, index)

	#apply alpha and beta for the two disctionary values
	query_dict.update((key, alpha * value) for key, value in query_dict.items())
	words_dict.update((key, beta * (value/k)) for key, value in words_dict.items())

	updated_query_dict = words_dict #words_dict is bigger than query dict, much more efficient to only update the few query words
	#variable is not copied, but referenciated. 

	for elem in query_dict:
		if elem in updated_query_dict:
			updated_query_dict[elem] = updated_query_dict[elem] + query_dict[elem]
		else:
			updated_query_dict[elem] = query_dict[elem]
	updated_query_dict = dict(sorted(updated_query_dict.items(),
						   key=lambda item: item[1],
						   reverse=True))
	#now we prune the dict
	return updated_query_dict

def remake_query(R, updated_query_dict):
	pruned_dict = prune_dict(R, updated_query_dict) #reduce dictionary, taking R more weighted words, we suppose that the dictionary is sorted decreasingly by weight

	#now we make the list for new query from dictionary
	remaked_query = []
	for item in pruned_dict.items():
		if item[1] != 0:
			remaked_query.append(str(item[0])+"^"+str(round(item[1],4))) #we round to avoid big numbers
	
	return remaked_query


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--index', default=None, help='Index to search')
	parser.add_argument('--k', default=10, type=int, help='Number of documents')
	parser.add_argument('--nrounds', default=5, type = int, help='Number of times we apply Rocchio Rule')
	parser.add_argument('--alpha', default=1, type = float, help='Alpha weight')
	parser.add_argument('--beta', default=1, type = float, help='Beta weight')
	parser.add_argument('--R', default=10, type = int, help='Maximum number of terms in the recomputed query')
	parser.add_argument('--query', default=None, nargs=argparse.REMAINDER, help='List of words to search')


	args = parser.parse_args()
	index = args.index
	k = args.k
	nrounds = args.nrounds
	alpha = args.alpha
	beta = args.beta
	R = args.R
	query = args.query
	print (query)

	client = Elasticsearch(timeout=10000)
	s = Search(using=client, index=index)
	
	try:
		k_doc_result = get_docs_from_query(k, query, s)
		for i in range(0, nrounds):
			updated_query_dict = compute_rocchio(alpha, beta, index, k, k_doc_result, query, client)
			new_query = remake_query(R, updated_query_dict) #from dictionary to query
			new_k_doc_result = get_docs_from_query(k, new_query, s)
			print (new_query)
			if(new_k_doc_result.hits.total['value']!=0): # the new AND query does not find a document
				query = new_query
				k_doc_result = new_k_doc_result
			else:
				print ("No more found documents with specific query")
				break
		print ("The last document query response collection which was not empty:")
		print ("Documents found:", k_doc_result.hits.total['value'])
		print(k_doc_result) #print last found document collection with not empty query
	except NotFoundError:
		print(f'Index {index} does not exist')
