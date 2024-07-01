import os
import math
import torch
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models import KeyedVectors, Word2Vec
import gensim

root_path = os.path.join(os.path.dirname(__file__), "../../")
model_path = f'{root_path}/src/models/text_embedding/GoogleNews-vectors-negative300.bin'
pretrained_model = KeyedVectors.load_word2vec_format(model_path, binary=True)




class Word2vecModel():
    def __init__(self) -> None:
        # self.model = Word2Vec.load("word2vec.model")
        pass
    
    def train(self):
        pretrained_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        capec = pd.read_csv(f"{root_path}/data/raw/capec.csv")
        cwe = pd.read_csv(f"{root_path}/data/raw/cwe.csv")
        cve = pd.read_csv(f"{root_path}/data/raw/cve.csv")
        capec_descriptions = [x for x in capec["Description"].tolist() if isinstance(x, str)] 
        cwe_descriptions   = [x for x in cwe["Description"].tolist() if isinstance(x, str)] 
        cve_descriptions = [x for x in cve["Description"].tolist() if isinstance(x, str)] 
        
        new_corpus = capec_descriptions + cwe_descriptions + cve_descriptions
        processed_new_corpus = [simple_preprocess(doc) for doc in new_corpus]
        
        new_model = Word2Vec(vector_size=pretrained_model.vector_size, min_count=1)
        new_model.build_vocab(processed_new_corpus)
        
        new_model.build_vocab([list(pretrained_model.key_to_index.keys())], update=True)
        new_model.train(processed_new_corpus, total_examples=len(processed_new_corpus), epochs=5)
        new_model.save("word2vec.model")
        
        similar_words = new_model.wv.most_similar('machine', topn=5)
        print(similar_words)
        
    
    def sentence_to_word2vec(self, sentence):
        words = sentence.lower().split()[:100]
        word_vectors = [pretrained_model[word].tolist() for word in words if word in pretrained_model]
        while len(word_vectors) < 100:
            word_vectors.append(np.zeros(300).tolist())
        return word_vectors


if __name__ == "__main__":
    Word2vecModel().train()