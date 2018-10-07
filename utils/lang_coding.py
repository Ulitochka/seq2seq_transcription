import sentencepiece as spm

from config import global_config

SOS_token = global_config.SOS_token
EOS_token = global_config.EOS_token


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class BPE:
    def __init__(self, bpe_model):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(bpe_model)

    def encode_ids(self, text):
        return self.sp.EncodeAsIds(text)

    def encode_pieces(self, text):
        return self.sp.EncodeAsPieces(text)

    def decode_ids(self, ids_massive):
        return self.sp.DecodeIds(ids_massive)

    def decode_pieces(self, p_massive):
        return [self.sp.DecodePieces(p_massive)]
