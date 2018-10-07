import itertools

import seaborn as sns
import matplotlib.pyplot as plt

from config import global_config
from utils.tokenizer import Tokenizer
from utils import tools


class VocabularyWorker:
    def __init__(self):

        self.tokenizer = Tokenizer()

        self.data_path = global_config.project_path + '/data/transcriptions'

        self.vocabularies = {
            "voc_l_1": [],
            "voc_l_2": []
        }

        self.correct_tokenizing = {
            "pairs_incorrect": 0,
            "pairs_correct": 0
        }

        data1, data2 = itertools.tee(self.data_loading(), 2)

        self.create_voc(data1)
        self.find_s_freq_th(data2)

        for el in self.correct_tokenizing:
            print(el, self.correct_tokenizing[el])

        for v in self.vocabularies:
            self.find_w_freq_th(self.vocabularies[v], v)

    def data_loading(self):
        with open(self.data_path, 'r') as file:
            data = [p.strip() for p in file.readlines()]
            pairs = [el.split('\t') for el in data]
            pairs_norm = [(self.tokenizer.tokenize(text=el[0], text_type='non_ph'),
                           self.tokenizer.tokenize(text=el[1], text_type='ph')) for el in pairs]
            for el in pairs_norm:

                for w_l_1 in el[0]:
                    self.vocabularies["voc_l_1"].append(w_l_1)
                for w_l_2 in el[1]:
                    self.vocabularies["voc_l_2"].append(w_l_2)

                if len(el[0]) != len(el[1]):
                    self.correct_tokenizing["pairs_incorrect"] += 1
                    # print(el, len(el[0]), len(el[1]))
                else:
                    self.correct_tokenizing["pairs_correct"] += 1
                    yield el

    def find_s_freq_th(self, data):
        data_len = [lengths for pair in data for lengths in [len(pair[0]), len(pair[1])]]
        print('max_sent_len:', max(data_len))
        sns.set(color_codes=True)
        sns.distplot(data_len, kde=False)
        plt.show()

    def find_w_freq_th(self, data, v_name):
        """
        Статистика для поиска оптимальной длины последовательности.
        :param data:
        :param v_name:
        :return:
        """
        print('*' * 100)
        print('count unique elements in %s:' % (v_name,), len(set(data)))
        len_voc = [len(el) for el in set(data)]
        print('max token len in %s' % (v_name,), max(len_voc))
        sns.set(color_codes=True)
        sns.distplot(len_voc, kde=False)
        plt.savefig(global_config.project_path + '/data/w_fig_v_%s' % (v_name,))

    def create_voc(self, data):
        temp_d = dict()
        data = [(pair[0][index], pair[1][index]) for pair in data for index in range(len(pair[0]))]
        for el in data:
            temp_d[el[0]] = temp_d.setdefault(el[0], []) + [el[1]]
        print('voc size:', len(temp_d))

        phone_dict = dict()
        for el in temp_d:
            phone_dict[el] = sorted(set(temp_d[el]))

        with open(global_config.project_path + '/data/few_transc.txt', 'w') as file:
            for k in temp_d:
                unique_ph = set(temp_d[k])
                if len(unique_ph) > 1:
                    file.write(k + ' => ' + str(unique_ph) + '\n')

        tools.save_binary(phone_dict, global_config.project_path + '/data/phone_dict.pkl')


if __name__ == '__main__':
    vw = VocabularyWorker()
