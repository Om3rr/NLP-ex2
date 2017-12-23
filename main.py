from nltk.corpus import dependency_treebank
from nltk.parse import DependencyGraph
from constants import *
from math import ceil
from scipy.sparse import csr_matrix
import numpy as np
import networkx as nx
from helper import EZSparseVector

def add_to_dict(key, value, d):
    if(key not in d):
        d[key] = {}
    if(value not in d[key]):
        d[key][value] = 0
    d[key][value] += 1

def build_words_graph(data):
    dWord = {}
    words = set()
    dPOS = {}
    POSS = set()
    for dg in data:
        for (head, headPOS),rel, (dep, depPOS) in dg.triples():
            add_to_dict(head,dep, dWord)
            add_to_dict(headPOS,depPOS, dPOS)
            words.add(head)
            words.add(dep)
            POSS.add(headPOS)
            POSS.add(depPOS)

    return dWord, dPOS, words, POSS

def getStringToIdx(words, tags):
    words = {word : idx for idx, word in enumerate(words)}
    tags = {tag : idx for idx, tag in enumerate(tags)}
    lenWords = len(words)
    lenTags = len(tags)
    def stringToIdx(elem, type):
        if(type == WORD):
            return words[elem]
        else:
            return tags[elem]

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




def twoHotVector(word1 : str, word2 : str, pos1 : str, pos2 : str, currentTheta, features):
    hot1, hot2 = features(word1, word2, WORD), features(pos1, pos2, TAG)
    tooHot = EZSparseVector([1,1], [hot1, hot2])
    return currentTheta.dot(tooHot)


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
            weight = twoHotVector(word1, word2, pos1, pos2, theta, strToIdx)
            G.add_edge(i,j,weight=-weight) # the algorithm is the minimal..
    G = nx.minimum_spanning_tree(G)
    return nxToVector(G, strToIdx)


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

def main():
    dWords, dPos, words, POSS = build_words_graph(train_data)
    stringToIdx, lenWords, lenTags = getStringToIdx(words, POSS)
    theta = EZSparseVector([], [])
    thetas = [theta]
    for i,sentence in enumerate(train_data):
        t = dgToVector(sentence, stringToIdx)
        t_tag = buildGraph(sentence, theta, stringToIdx)
        theta = theta - (t - t_tag)
        thetas.append(theta)


if __name__ == '__main__':
    train_data, test_data = init()
    main()