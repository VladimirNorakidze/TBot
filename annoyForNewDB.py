import numpy as np
import pandas as pd

from annoy import AnnoyIndex
from gensim.models import KeyedVectors, Word2Vec

import re, sys
import pymorphy2
from functools import lru_cache
morph = pymorphy2.MorphAnalyzer()

@lru_cache(maxsize=100000) # с кешом!
def get_normal_form (i):
    return morph.normal_forms(i)[0]

def normalize_text(x):
    return ' '.join([get_normal_form(i) for i in re.findall('\w+', x)])

def create_data(data_file):
    data = pd.read_csv(data_file)
    data = data[['title', 'product_id']]  # product_id is here for fun
    
    COLS = ['title']
    for col in COLS:
        print(col, '- normalization procesing...')
        get_normal_form.cache_clear()
        data[col] = data[col].apply(lambda x: normalize_text(x))

    print('All titles to list....')
    data['title'] = data['title'].apply(lambda x: x.split(' '))
    print('Finish!')
    
    return data

    
def create_annoy(data_file, model_dir, model_file):
    data = create_data(data_file)
    
    model = KeyedVectors.load(model_dir+model_file)
    model.wv.vector_size
    
    matrix = []
    for tit in data['title']:
        #print(tit, prod_id, img_link)

        list_v = []
        for word in tit:
            try:
                list_v.append(model.wv[word])
            except KeyError:
                list_v.append([0]*300)

        wrap_list = []
        if list_v:
            wrap_list.append(np.mean(np.array(list_v), axis=0))
        else:
            wrap_list.append(np.array([0]*300))

        matrix.append(np.mean(wrap_list,axis=0))
    print('Matrix was created ', matrix[0].shape)
    
    NUM_TREES = 15
    VEC_SIZE_EMB = model.wv.vector_size
    mapId2wordHash = {}
    index_title_emb = AnnoyIndex(VEC_SIZE_EMB)

    print('Build annoy base...')
    for word_hash, counter in zip(matrix, range(len(matrix))):
        index_title_emb.add_item(counter, word_hash)
        mapId2wordHash[counter] = word_hash

    index_title_emb.build(NUM_TREES)
    print('Finish!')
    
    FILENAME = model_dir+'annoy_'+str(NUM_TREES)
    index_title_emb.save(FILENAME)
    
if __name__ == "__main__":
    # argv: 1- data_filename, 2 - dir where w2v mode, 3 - w2v filename
    # create_annoy('data/dataset_sample.csv', 'ModelsOnBigBase/3/', 'w2v.model') 
    create_annoy(sys.argv[1], sys.argv[2], sys.argv[3])  