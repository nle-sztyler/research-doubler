import numpy as np
from collections import namedtuple


def get_known_triples(data: namedtuple, graph_properties: namedtuple):
    triples = set()

    vertex_indexer = graph_properties.vertex_indexer
    relation_indexer = graph_properties.relation_indexer

    triples = triples.union(extract_triples(data.triples_train, vertex_indexer, relation_indexer))
    triples = triples.union(extract_triples(data.triples_validation, vertex_indexer, relation_indexer))

    return triples


def extract_triples(data: np.array, vertex_indexer: dict, relation_indexer: dict):
    triples = [(vertex_indexer[triple[0]], relation_indexer[triple[1]], vertex_indexer[triple[2]])
               for triple in data
               if triple[0] in vertex_indexer and triple[1] in relation_indexer and triple[2] in vertex_indexer]

    return set(triples)


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def pk(preds, preds_neg, k=50):
    result = [(1, entry) for entry in preds] + [(0, entry) for entry in preds_neg]
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
    k = min([k, len(sorted_result)])
    positive = [1 for entry in sorted_result[:k] if entry[0] == 1]

    return float(len(positive)) / k
