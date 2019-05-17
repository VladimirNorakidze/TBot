import numpy as np
import pandas as pd

from annoy import AnnoyIndex
from gensim.models import KeyedVectors, Word2Vec

import re
import pymorphy2
from functools import lru_cache
morph = pymorphy2.MorphAnalyzer()


@lru_cache(maxsize=100000)
def get_normal_form (i):
    return morph.normal_forms(i)[0]


def normalize_text(x):
    return ' '.join([get_normal_form(i) for i in re.findall('\w+', x)])


class VisBotTextBrain:
    
    def __init__(self, model_file, annoy_file):
        self.model = KeyedVectors.load(model_file)  # Word2Vec.load
        self.annoy = AnnoyIndex(self.model.wv.vector_size)
        self.annoy.load(annoy_file)

    def run(self, request):
        """
        :param: request: str value
        :return: list of indexes in DB
        """
        request_list = normalize_text(request).split(' ')
        
        if (len(request_list)==1)and(request_list[0]==''):
            raise ValueError('Uncorrect request')
        
        vect_repr = []
        for word in request_list:
            try:
                vect_repr.append(self.model.wv[word])
            except KeyError:
                vect_repr.append([0]*300)

        if vect_repr:
            vect_repr = np.mean(np.array(vect_repr), axis=0)
        else:
            vect_repr = np.array([0]*300)
    
        self.request = request
        self.request_vector = vect_repr
    
        return self.annoy.get_nns_by_vector(vect_repr, n=10)


def main(request):
    model_file = "w2v.model"
    annoy_file = "annoy15"
    vbtb = VisBotTextBrain(model_file, annoy_file)
    try:
        vbtb.run(request)
    except ValueError:
        print('Incorrect request, bro. Type another string')