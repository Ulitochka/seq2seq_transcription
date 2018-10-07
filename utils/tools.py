import pickle
import os
import time
import logging
import uuid
import math

import torch

from utils.lang_coding import Lang
from config import global_config
from utils.tokenizer import Tokenizer


tokenizer = Tokenizer()


def init_logging():
    fmt = logging.Formatter('%(asctime)-15s %(message)s')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_dir_name = global_config.project_path + '/log/'
    log_file_name = time.strftime("%Y_%m_%d-%H_%M_%S-") + str(uuid.uuid4())[:8] + '.txt'
    logging.info('Logging to {}'.format(log_file_name))
    logfile = logging.FileHandler(os.path.join(log_dir_name, log_file_name), 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)


def readLangs(lang1, lang2, reverse=False, pairs=False, data_path=None):
    if not pairs:
        with open(data_path, 'r') as file:
            data = [p.strip() for p in file.readlines()]
            pairs = [el.split('\t') for el in data]
            pairs = [(tokenizer.tokenize(text=el[0], text_type='non_ph'),
                      tokenizer.tokenize(text=el[1], text_type='ph')) for el in pairs]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepareData(input_lang, output_lang, pairs, reverse=False):
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def filterPairs(pairs, max_len):
    return [pair for pair in pairs if len(pair[0]) <= max_len and len(pair[1]) <= max_len]


def tensotFromPair(lang, text, EOS_token, device):
    if lang:
        indexes = [lang.word2index.get(word, 2) for word in text] + [EOS_token]
    else:
        indexes = text + [EOS_token]
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(lang1, lang2, pair, EOS_token, device):
    input_tensor = tensotFromPair(lang1, pair[0], EOS_token, device)
    output_tensor = tensotFromPair(lang2, pair[1], EOS_token, device)
    return input_tensor, output_tensor


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def save_binary(data, file_name):
    """
    Save data in binary format.
    :param data:
    :param file_name:
    :return:
    """

    with open(file_name, 'wb') as file:
        pickle.dump(data, file, protocol=4)


def load_bin_data(path_to_data):
    """
    Load binary data.
    :return:
    """

    with open(path_to_data, 'rb') as f:
        data = pickle.load(f)
    return data
