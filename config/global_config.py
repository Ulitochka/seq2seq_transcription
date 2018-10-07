import os

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
bpe_model = project_path + '/models/bpe/'
max_sent_length = 16  # max = 15
teacher_forcing_ratio = 0.5

BPE = False

SOS_token = 0
EOS_token = 1

hidden_size = 1024
clip_grad = 5.0
n_layers = 2
learning_rate = 0.0001
dropout = 0.5
iterations = 170000
