from __future__ import unicode_literals, print_function, division
from io import open
import json
from time import time
import random

import torch.nn as nn
from torch import optim

from utils.tools import *
from config import global_config
from utils.lang_coding import BPE
from models.seq2seq.model import EncoderRNN, AttnDecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self):

        data = load_bin_data(global_config.project_path + '/data/train_test_pairs.pkl')
        self.X_train = data['X_train']

        if global_config.BPE:

            self.bpe_indexer_lang1 = BPE(bpe_model=global_config.project_path + '/models/bpe/bpe_lang1.model')
            self.bpe_indexer_lang2 = BPE(bpe_model=global_config.project_path + '/models/bpe/bpe_lang2.model')

            self.X_train = [(
                self.bpe_indexer_lang1.encode_ids(' '.join(s[0])),
                self.bpe_indexer_lang2.encode_ids(' '.join(s[1]))
                ) for s in self.X_train]

            self.max_sent_length = max([max([len(el[0]), len(el[1])]) for el in self.X_train])
            self.SOS_token = self.bpe_indexer_lang1.sp.PieceToId('</s>')
            self.EOS_token = self.bpe_indexer_lang1.sp.PieceToId('<s>')

            self.input_voc_size = self.bpe_indexer_lang1.sp.__len__()
            self.output_voc_size = self.bpe_indexer_lang2.sp.__len__()

            self.input_lang = None
            self.output_lang = None

        else:

            input_lang, output_lang, pairs = readLangs('G', 'P', reverse=False, pairs=self.X_train)
            logging.info('count pairs before filter: ' + str(len(pairs)))
            pairs = filterPairs(pairs, global_config.max_sent_length)
            logging.info('count pairs after filter: ' + str(len(pairs)))

            self.input_lang, self.output_lang, pairs = prepareData(input_lang, output_lang, pairs, reverse=False)

            save_binary(
                {
                    "input_lang": input_lang,
                    "output_lang": output_lang
                }, global_config.project_path + '/data/vocs.pkl')

            self.max_sent_length = global_config.max_sent_length
            self.SOS_token = global_config.SOS_token
            self.EOS_token = global_config.EOS_token

            self.input_voc_size = self.input_lang.n_words
            self.output_voc_size = self.output_lang.n_words

        self.teacher_forcing_ratio = global_config.teacher_forcing_ratio

    def train_loop(self,
                   input_tensor,
                   target_tensor,
                   encoder,
                   decoder,
                   encoder_optimizer,
                   decoder_optimizer,
                   criterion):
        """

        :param input_tensor:
        :param target_tensor:
        :param encoder:
        :param decoder:
        :param encoder_optimizer:
        :param decoder_optimizer:
        :param criterion:
        :return:
        """

        encoder_hidden = encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.max_sent_length, encoder.hidden_size, device=device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[self.SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                            decoder_hidden,
                                                                            encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                            decoder_hidden,
                                                                            encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == self.EOS_token:
                    break

        loss.backward()

        torch.nn.utils.clip_grad_norm(encoder.parameters(), global_config.clip_grad)
        torch.nn.utils.clip_grad_norm(decoder.parameters(), global_config.clip_grad)

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def trainIters(self,
                   *,
                   encoder,
                   decoder,
                   n_iters,
                   print_every=1000,
                   learning_rate=0.0001):
        """

        :param encoder:
        :param decoder:
        :param n_iters:
        :param print_every:
        :param learning_rate:
        :return:
        """

        start = time.time()
        print_loss_total = 0  # Reset every print_every

        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

        training_pairs = [
            tensorsFromPair(
                lang1=self.input_lang,
                lang2=self.output_lang,
                pair=random.choice(self.X_train),
                EOS_token=self.EOS_token,
                device=device
            )
            for i in range(n_iters)]
        criterion = nn.NLLLoss()

        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = self.train_loop(input_tensor,
                                   target_tensor,
                                   encoder,
                                   decoder,
                                   encoder_optimizer,
                                   decoder_optimizer,
                                   criterion)

            print_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                logging.info('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                                    iter, iter / n_iters * 100, print_loss_avg))

        model_id = str(time.time())

        with open(global_config.project_path + '/models/bin/model_seq2seq_encoder_%s.pkl' % (model_id,), 'wb') as f:
            torch.save(encoder, f)

        with open(global_config.project_path + '/models/bin/model_seq2seq_decoder_%s.pkl' % (model_id,), 'wb') as f:
            torch.save(decoder, f)

        with open(global_config.project_path + '/models/bin/config_%s.json' % (model_id,), 'w') as outfile:
            outfile.write(json.dumps(
                {
                    "max_sent_len": self.max_sent_length,
                    "teacher_forcing_ratio": global_config.teacher_forcing_ratio,
                    "hidden_size": global_config.hidden_size,
                    "clip_grad": global_config.clip_grad,
                    "n_layers": global_config.n_layers,
                    "learning_rate": global_config.learning_rate,
                    "dropout": global_config.dropout,
                    "BPE": global_config.BPE,
                    "iterations": global_config.iterations
                },
                sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': ')))

    def train(self):
        """

        :return:
        """

        hidden_size = global_config.hidden_size
        encoder = EncoderRNN(input_size=self.input_voc_size,
                             hidden_size=hidden_size,
                             n_layers=global_config.n_layers).to(device)

        attn_decoder = AttnDecoderRNN(n_layers=global_config.n_layers,
                                      hidden_size=hidden_size,
                                      output_size=self.output_voc_size,
                                      dropout_p=global_config.dropout,
                                      max_length=self.max_sent_length).to(device)

        self.trainIters(encoder=encoder,
                        decoder=attn_decoder,
                        n_iters=global_config.iterations,
                        learning_rate=global_config.learning_rate,
                        print_every=1000)


if __name__ == '__main__':
    init_logging()
    logging.info("Start")
    tr = Trainer()
    tr.train()
