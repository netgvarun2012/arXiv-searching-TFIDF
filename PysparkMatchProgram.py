#!/usr/bin/env python3
import json
import re
import sys
from pyspark.sql import SparkSession
from collections import Counter
import math
from pyspark import SparkContext

# Obtaining a connection to a Spark cluster, which will be used to create RDDs.
sc  = SparkContext.getOrCreate()
spark = SparkSession(sc)

# Reading a txt file and converting it into a list of stop words.
with open(sys.argv[1],'r') as file:
    stopwords = file.read().splitlines()

# Reading the data json file provided as input.
dataset_json = sc.textFile(sys.argv[2])
dataset = dataset_json.map(lambda x: json.loads(x))

# Creating a RDD to work on abstracts.
abstract = dataset.map(lambda x: (x['id'],x['abstract']))

# storing the number of asbtracts.
N = abstract.count()        

# Creating a RDD to work on titles.
title_rdd = dataset.map(lambda x: (x['id'],x['title']))

def compute_TFIDF(abstract_rdd,stopwords,title_rdd,flag):
    """ 
    A function writen to compute TDIDF for both abstracts and titles.
    Variable flag in input indicates if the function is being called for Abstracts or for Titles.  
    """

    global pairs_DF

    ############## Computing DF(Document Frequencies) ################

    if(flag == 0):                                                # calculating TFIDF for abstracts
        rdd_for_docu_freq = abstract_rdd
        rdd_for_term_freq = abstract_rdd

        # getting rid of the stop words.
        wo_stopwords = rdd_for_docu_freq.mapValues(lambda x: ' '.join(word for word in x.split() if word.lower() not in stopwords))

        # removing punctuations (Note the use of the 'set' function to get rid of duplicates since we are calculating document frequency)
        words = wo_stopwords.mapValues(lambda x: list(set(re.findall(r'[^!?.,-;\$\n()\\ ]+',x.lower()))))

        # obtaining word frequencies within each document
        pairs = words.mapValues(lambda x: list(Counter(x).items()))

        # summing up the values for each word across document, this is done by updating the values in the original RDD with the counts from the previous step.
        counts = pairs.flatMap(lambda x: x[1]).reduceByKey(lambda a, b: a + b).collectAsMap()
        pairs_DF = pairs.map(lambda x: (x[0], [(k, counts[k]) for k, v in x[1]]))

    if(flag == 1):                                                # calculating TFIDF for titles
        rdd_for_docu_freq = abstract_rdd
        rdd_for_term_freq = title_rdd

    ################## Computing TF(Term Frequencies) ####################

    # removing punctuations from the rdd (abstract or title depending upon the flag value passed).
    wo_stopwords_TF = rdd_for_term_freq.mapValues(lambda x: ' '.join(word for word in x.split() if word.lower() not in stopwords))

    # preparing count RDD for calculating term frequency (set function is omitted now as we want the frequency)
    words_TF = wo_stopwords_TF.mapValues(lambda x: list(re.findall(r'[^!?.,-;\n()\\ ]+',x.lower())))
  
    # getting the Term frequencies.
    pairs_TF = words_TF.mapValues(lambda x: list(Counter(x).items()))
 
    ############### Computing TFIDF ##################

    # flattening both RDDs using flatMap, then joining them based on the key and the first element of the tuple list.
 
    rdd1_flat = pairs_TF.flatMap(lambda x: [((x[0], y[0]), y[1]) for y in x[1]])
    rdd2_flat = pairs_DF.flatMap(lambda x: [((x[0], y[0]), y[1]) for y in x[1]])

    joined = rdd1_flat.join(rdd2_flat)

    # joined RDD is then grouped by the key using the groupByKey transformation.
    result2 = joined.map(lambda x: (x[0][0], (x[0][1], x[1][0], x[1][1]))).groupByKey().mapValues(list)

    # calculating the actual TFIDF value by using the formula.
    result3 = result2.map(lambda x: (x[0], [(k, (1+ math.log10(v1)) * math.log10(N/v2)) for k, v1, v2 in x[1]]))

    # normalizing the TF-IDF values by dividing with the sum of squares of the TF-IDF of all the words for each doc.
    result4 = result3.map(lambda x: (x[0], [(k, v / math.sqrt(sum(y[1] for y in x[1]))) for k, v in x[1]]))
    
    # returning the TFIDF value calculated.
    return result4

def multiply_common_keys(pair):
    (key1, values1), (key2, values2) = pair
    values1_dict = dict(values1)
    values2_dict = dict(values2)
    common_keys = set(values1_dict.keys()) & set(values2_dict.keys())
    result = [(k, values1_dict[k] * values2_dict[k]) for k in common_keys]
    return ((key2, key1), result)


def remove_first_element(pair):
    (key1, key2), values = pair
    result = [v for k, v in values]
    return ((key1, key2), sum(result))

# function call to calculate TFIDF value for each abstract.
abstracts_tfidf = compute_TFIDF(abstract,stopwords,title_rdd,0).map(lambda x: (x[0],x[1]))

# function call to calculate TFIDF value for each title.
titles_tfidf = compute_TFIDF(abstract,stopwords,title_rdd,1).map(lambda x: (x[0],x[1]))

# multiplying TFIDF for titles and abstracts for similarity score calculations
RDD3 = abstracts_tfidf.cartesian(titles_tfidf).map(multiply_common_keys)

# removing words from the tuples to keep only the similarity score.
RDD_wo_words = RDD3.map(remove_first_element)

# getting the top-ranked abstract for each query.
RDD_sim = RDD_wo_words.map(lambda x: (x[0][0], x)).reduceByKey(lambda x, y: x if x[1] > y[1] else y).map(lambda x: x[1])

# for each of the top-ranked abstract, calculating if its a 'hit or a 'miss and showing misses as a dataframe
count_mismatch = RDD_sim.filter(lambda x: x[0][0] != x[0][1]).map(lambda x: (x[0][0],x[0][1],x[1]))
df = count_mismatch.toDF(['Title id','Abstract id','CosineSimilarity'])
df.show()

# for each of the top-ranked abstract, calculating if its a 'hit or a 'miss and summing them up.
count = RDD_sim.filter(lambda x: x[0][0] == x[0][1]).map(lambda x: 1).sum()

# calculating the final accuracy by dividing by total number of abstracta.
acc_score = (count/N)*100

print('\nAccuracy score is:')
print("%0.00f%%" % (acc_score))
print('\n')

print("\nNumber of titles that are not correctly matching with their abstracts are {}.".format(count_mismatch.count()))
print('\n')
sc.stop()