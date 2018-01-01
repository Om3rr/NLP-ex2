from nltk.corpus import dependency_treebank
from nltk.parse import DependencyGraph
from constants import *
from math import ceil
from scipy.sparse import csr_matrix
import numpy as np
import networkx as nx
from helper import EZSparseVector

IDXFEATURESIZE = 0
AUGMENTED = True

def add_to_dict(key, value, d):
    if(key not in d):
        d[key] = {}
    if(value not in d[key]):
        d[key][value] = 0
    d[key][value] += 1

def build_words_graph(data):
    words = set()
    POSS = set()
    for dg in data:
        for (head, headPOS),rel, (dep, depPOS) in dg.triples():
            words.add(head)
            words.add(dep)
            POSS.add(headPOS)
            POSS.add(depPOS)

    return words, POSS

def getStringToIdx(words, tags):
    words = {word : idx for idx, word in enumerate(words)}
    tags = {tag : idx for idx, tag in enumerate(tags)}
    lenWords = len(words)
    lenTags = len(tags)
    IDXFEATURESIZE = lenWords**2 + lenTags**2
    def stringToIdx(elem, type):
        if(type == WORD):
            return words[elem] if elem in words else 0
        else:
            return tags[elem] if elem in tags else 0
    def doubleToIdx(elem1, elem2, type):
        a,b = stringToIdx(elem1, type), stringToIdx(elem2, type)
        if(type == WORD):
            return a*lenWords + b
        else:
            return lenWords**2 + a*lenTags + b
    return doubleToIdx, lenWords, lenTags


def getFeatureFunction(strToIdx, sizes):

    def elemsToFeature(x1,x2,type):
        feature = strToIdx(x1, type) * sizes[type] + strToIdx(x2, type)
        if(type == TAG):
            feature += sizes[WORD]**2
        return feature

    def featureFunction(dg : DependencyGraph):
        cols = set()
        for (head, headPOS),rel, (dep, depPOS) in dg.triples():
            cols.add(elemsToFeature(head, dep, WORD))
            cols.add(elemsToFeature(headPOS, depPOS, TAG))
        return csr_matrix(([1]*len(cols), ([0]*len(cols), list(cols))), shape=(1, sizes['vector']))
    return featureFunction




def twoHotVector(word1 : str, word2 : str, pos1 : str, pos2 : str, features):
    hot1, hot2 = features(word1, word2, WORD), features(pos1, pos2, TAG)
    tooHot = EZSparseVector([1,1], [hot1, hot2])
    return tooHot

def diffToFeature(diff):
    return IDXFEATURESIZE+diff

def buildGraph(sentence, theta, strToIdx):
    G = nx.empty_graph()
    sentenceLen = len(sentence.nodes)
    for k, node in list(sentence.nodes.items())[1:]:
        G.add_node(k, node)
    for i in range(1,sentenceLen):
        word1,pos1 = G.node[i]['word'], G.node[i]['tag']
        for j in range(1, sentenceLen):
            if(i==j):
                continue
            word2, pos2 = G.node[j]['word'], G.node[j]['tag']
            hotVector = twoHotVector(word1, word2, pos1, pos2, strToIdx)
            # if abs(i-j) < 5: # AUGMENTED
            #     hotVector += EZSparseVector([1], [diffToFeature(abs(i - j))])
            weight = theta.dot(hotVector)
            G.add_edge(i,j,weight=-weight) # the algorithm is the minimal..
    return G


def nxToVector(G, strToIdx):
    posses = []
    for e in G.edges_iter():
        fromV, toV = e
        fromWord, fromTag = G.node[fromV]['word'], G.node[fromV]['tag']
        toWord, toTag = G.node[toV]['word'], G.node[toV]['tag']
        posses += [strToIdx(fromWord, toWord, WORD), strToIdx(fromTag, toTag, TAG)]
    return EZSparseVector([1]*len(posses), posses)

def dgToVector(dg, strToIdx):
    posses = []
    for (fromW, fromV), rel, (toW, toV) in dg.triples():
        posses += [strToIdx(fromW, toW, WORD), strToIdx(fromV, toV, TAG)]
    return EZSparseVector([1] * len(posses), posses)





def getMST(sentence : DependencyGraph, featureFunction, w : csr_matrix):
    G = nx.empty_graph()
    for node in enumerate(sentence.nodes):
        G.add_node(node)




def getRandomSentence():
    return train_data[0]

def init():
    sentences = dependency_treebank.parsed_sents()
    cut_idx = ceil(len(sentences) * (1 - TEST_PERCENT))
    train_data = sentences[:cut_idx]
    test_data = sentences[cut_idx:]
    return train_data, test_data

import random, time

def trainModel(train_data, stringToIdx):
    theta = EZSparseVector([], [])
    summedTheta = EZSparseVector([], [])
    diffs = []
    for j in range(ITERATIONS):
        random.shuffle(train_data)
        for i, sentence in enumerate(train_data):
            if(i%100==0):
                print(i)
            t = dgToVector(sentence, stringToIdx)
            t_tag = nxToVector(nx.minimum_spanning_tree(buildGraph(sentence, theta, stringToIdx)), stringToIdx)
            diff = (t - t_tag)
            theta = theta + diff
            diffs.append(diff)
    return getW(diffs, len(train_data)*ITERATIONS)

def dgToEdges(sentence):
    return [(w1,w2) for (w1, t1), l, (w2, t2) in sentence.triples()]


def getW(diffs, N):
    summedTheta = EZSparseVector([],[])
    # I prefer to calculate the diffs because it will be much faster than sum everything up...
    for i,diff in enumerate(diffs):
        c = len(diffs) - i
        diff*=c
        summedTheta += diff
    summedTheta /= N
    return summedTheta

partialPercent = []
percentage = []
def testModel(test_data, w, stringToIdx):
    overall, count = 0,0
    for testSetence in test_data:
        realEdges = set(dgToEdges(testSetence))
        G = nx.minimum_spanning_tree(buildGraph(testSetence, w, stringToIdx))
        Gnode = G.node
        maxEdges = set([(Gnode[j]['word'], Gnode[i]['word']) for i,j in G.edges()])
        count += len(maxEdges.intersection(realEdges))
        overall += len(maxEdges)
        partialPercent.append(len(maxEdges.intersection(realEdges)) / len(maxEdges))
        percentage.append(count/overall)
        print("{}% success rate, {} from {} in this sentence".format(count/overall, len(maxEdges.intersection(realEdges)), len(maxEdges)))

import matplotlib.pyplot as plt
def main():
    words, POSS = build_words_graph(train_data)
    stringToIdx, lenWords, lenTags = getStringToIdx(words, POSS)
    w = trainModel(train_data, stringToIdx)
    testModel(test_data, w, stringToIdx)
    print(partialPercent)
    print(percentage)
    partial, = plt.plot(range(len(partialPercent)), partialPercent, 'b', label='Partial Percentage')
    cumulative, = plt.plot(range(len(partialPercent)), percentage, 'g', label='Cumulative Percentage')
    plt.legend(handles=[partial, cumulative])
    plt.show()






if __name__ == '__main__':
    train_data, test_data = init()
    main()