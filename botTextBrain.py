import numpy as np
import pandas as pd

from annoy import AnnoyIndex
from gensim.models import KeyedVectors, Word2Vec

from annoy import AnnoyIndex
from gensim.models import KeyedVectors, Word2Vec

import re
import pymorphy2
from functools import lru_cache

morph = pymorphy2.MorphAnalyzer()


@lru_cache(maxsize=100000)
def get_normal_form (i):  # Функция преведения слова к нормальной форме
    return morph.normal_forms(i)[0]


def normalize_text(x):  # Функция, приводящая слова в строке к нормальной форме
    return ' '.join([get_normal_form(i) for i in re.findall('\w+', x)])


class VisBotTextBrain:

    def __init__(self, model_file, annoy_file):
        print("Loading w2v model...")
        self.model = KeyedVectors.load("./data/" + model_file)  # Word2Vec.load
        self.annoy = AnnoyIndex(self.model.wv.vector_size)
        print("Loading Annoy...")
        self.annoy.load("./data/" + annoy_file)

    def run(self, request=None):
        """
        :param: request: str value
        :return: list of indexes in DB
        """
        request_list = normalize_text(request).split(' ')

        if (len(request_list) == 1) and (request_list[0] == ''):
            raise ValueError('Incorrect request')

        vect_repr = self._get_vect_representation(request_list)

        self.request = request
        self.request_vector = vect_repr

        return self.annoy.get_nns_by_vector(vect_repr, n=100)

    def _get_vect_representation(self, request_list):
        vect_repr = []
        for word in request_list:
            try:
                vect_repr.append(self.model.wv[word])
            except KeyError:
                vect_repr.append([0] * 300)

        if vect_repr:
            vect_repr = np.mean(np.array(vect_repr), axis=0)
        else:
            vect_repr = np.array([0] * 300)

        return vect_repr


annoy_file = "annoy15"
model_file = "w2vsize300window2.model"
vbtb = VisBotTextBrain(model_file, annoy_file)


def main(request=None):
    if request:
        try:
            return vbtb.run(request=request)
        except ValueError:
            print('Incorrect request, bro. Type another string')
