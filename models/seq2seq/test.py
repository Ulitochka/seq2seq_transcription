from tqdm import tqdm
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

from utils.tools import *
from utils.lang_coding import BPE
from config import global_config


class ModelTest:
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = load_bin_data(global_config.project_path + '/data/train_test_pairs.pkl')
        self.X_test = data['X_test']

        if global_config.BPE:

            self.bpe_indexer_lang1 = BPE(bpe_model=global_config.project_path + '/models/bpe/bpe_lang1.model')
            self.bpe_indexer_lang2 = BPE(bpe_model=global_config.project_path + '/models/bpe/bpe_lang2.model')

            self.X_test = [(
                self.bpe_indexer_lang1.encode_ids(' '.join(s[0])),
                self.bpe_indexer_lang2.encode_ids(' '.join(s[1]))
                ) for s in self.X_test]

            self.max_sent_length = max([max([len(el[0]), len(el[1])]) for el in self.X_test])
            self.SOS_token = self.bpe_indexer_lang1.sp.PieceToId('</s>')
            self.EOS_token = self.bpe_indexer_lang1.sp.PieceToId('<s>')

            self.input_voc_size = self.bpe_indexer_lang1.sp.__len__()
            self.output_voc_size = self.bpe_indexer_lang2.sp.__len__()

            self.input_lang = None
            self.output_lang = None

        else:
            vocs = load_bin_data(global_config.project_path + '/data/vocs.pkl')
            self.input_lang = vocs["input_lang"]
            self.output_lang = vocs["output_lang"]

            self.SOS_token = global_config.SOS_token
            self.EOS_token = global_config.EOS_token

            self.max_sent_length = global_config.max_sent_length

        self.model_id = '1538732901.832973'

        model = self.model_loader()
        self.encoder = model['encoder']
        self.decoder = model['decoder']

        self.evaluate()

    def torch_loading(self, file):
        with open(global_config.project_path + file, 'rb') as f:
            return torch.load(f)

    def model_loader(self):
        models_parts = [
            ('encoder', '/models/bin/model_seq2seq_encoder_%s.pkl' % (self.model_id,)),
            ('decoder', '/models/bin/model_seq2seq_decoder_%s.pkl' % (self.model_id,))
        ]
        return {file[0]: self.torch_loading(file[-1]) for file in models_parts}

    def bleu_score(self, pred_pronunciation, test_pronunciation):
        smooth = SmoothingFunction().method1
        return sentence_bleu([pred_pronunciation], test_pronunciation, smoothing_function=smooth)

    def predict(self, encoder, decoder, text):
        """
        Генерация "перевода".
        :param encoder:
        :param decoder:
        :param pair:
        :return:
        """

        with torch.no_grad():

            input_tensor = tensotFromPair(self.input_lang, text, self.EOS_token, self.device)

            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.initHidden()
            encoder_outputs = torch.zeros(self.max_sent_length, encoder.hidden_size, device=self.device)
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[self.SOS_token]], device=self.device)  # SOS
            decoder_hidden = encoder_hidden

            decoded_words = []
            for di in range(self.max_sent_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                            decoder_hidden,
                                                                            encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == self.EOS_token:
                    break
                else:
                    if global_config.BPE:
                        decoded_words.append(topi.item())
                    else:
                        decoded_words.append(self.output_lang.index2word[topi.item()])
                decoder_input = topi.squeeze().detach()
            return decoded_words

    def evaluate(self):

        correct_decoded_words = 0
        correct_decoded_sentences = 0
        blue = []
        count_words = 0
        count_sentences = len(self.X_test)

        with open(global_config.project_path + '/results/test_pairs_%s.txt' % (self.model_id,), 'w') as file:
            for index in tqdm(range(len(self.X_test))):
                mark = 0

                decoded_sent = self.predict(self.encoder, self.decoder, self.X_test[index][0])
                true_sent = self.X_test[index][1]

                if global_config.BPE:
                    decoded_sent = self.bpe_indexer_lang2.decode_ids(decoded_sent).split(' ')
                    true_sent = self.bpe_indexer_lang2.decode_ids(true_sent).split(' ')

                if decoded_sent == true_sent:
                    correct_decoded_sentences += 1
                    mark = 1

                correct_decoded_words += len([element for element in decoded_sent if element in true_sent])
                count_words += len(true_sent)
                blue.append(self.bleu_score(decoded_sent, true_sent))

                file.write('mark= ' + str(mark) +
                           ' input= ' + ' '.join(true_sent) +
                           ' predict= ' + ' '.join(decoded_sent) + '\n')
        file.close()

        acc_per_word = correct_decoded_words / count_words
        acc_per_sentence = correct_decoded_sentences / count_sentences

        with open(global_config.project_path + '/results/model_%s.json' % (self.model_id,), 'w') as outfile:
                    outfile.write(json.dumps(
                        {
                            "acc_per_word": acc_per_word,
                            "acc_per_sentence": acc_per_sentence,
                            "blue": np.mean(blue)
                        },
                        sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': ')))


if __name__ == '__main__':
    ModelTest()
