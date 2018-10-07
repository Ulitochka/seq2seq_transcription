from sklearn.model_selection import train_test_split

from utils.tokenizer import Tokenizer
from utils.tools import *
from config import global_config


class BpeCorpora:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.data_path = global_config.project_path + '/data/transcriptions'

        self.lang_1 = []
        self.lang_2 = []

    def write2file(self, data, file_name):
        with open(global_config.project_path + '/models/bpe/%s.txt' % (file_name,), 'w') as file:
            for k in data:
                file.write(' '.join(k).strip() + '\n')
        file.close()

    def load_data(self):
        with open(self.data_path, 'r') as file:
            data = [p.strip() for p in file.readlines()]
            pairs = [el.split('\t') for el in data]
            pairs_norm = [(tokenizer.tokenize(text=el[0], text_type='non_ph'),
                           tokenizer.tokenize(text=el[1], text_type='ph')) for el in pairs]

            X_train, X_test = train_test_split(pairs_norm, test_size=0.1, random_state=1024)

            for p in X_train:
                self.lang_1.append(p[0])
                self.lang_2.append(p[1])

            x = set([t for s in X_test for t in s[0]])
            y = set([t for s in X_train for t in s[0]])

            print('voc train:', len(x))
            print('voc test:', len(y))
            print('voc all:', len(x.union(y)))
            print('diff', len(y.difference(x)))

            print(len(X_train), len(X_test))

            save_binary(
                {
                    "X_test": X_test,
                    "X_train": X_train
                }, global_config.project_path + '/data/train_test_pairs.pkl')

        self.write2file(self.lang_1, 'bpe_lang1')
        self.write2file(self.lang_2, 'bpe_lang2')


if __name__ == '__main__':
    bpe_corpora_creator = BpeCorpora()
    bpe_corpora_creator.load_data()
