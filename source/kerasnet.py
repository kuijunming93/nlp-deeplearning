import utilities
import os, sys

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences

try:
  import keras.backend as K
  if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU
    print("GPU")
except:
  pass

# some config
BATCH_SIZE = 64  # Batch size for training.
EPOCHS = 40  # Number of epochs to train for.
LATENT_DIM = 256  # Latent dimensionality of the encoding space.
NUM_SAMPLES = 10000  # Number of samples to train on.
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

# DATA PROCESSING
# reading raw files into memory for processing
memoryLine = {}
with open('./resources/movie_lines.txt', encoding='utf-8', errors='ignore') as f:
    _movieLines = f.read().split("\n")
    for _line in _movieLines:
        _element = _line.split(" +++$+++ ")
        if len(_element) == 5:
            memoryLine[_element[0]] = _element[4]

memoryConversation = []
with open('./resources/movie_conversations.txt', encoding='utf-8', errors='ignore') as f:
    _movieConversations = f.read().split("\n")
    for _conversation in _movieConversations:
        _element = _conversation.split(" +++$+++ ")[-1][1:-1].replace("'","").replace(" ","")
        memoryConversation.append(_element.split(","))

clean_questions = []
clean_answers = []
for conversation in memoryConversation:
    for i in range(len(conversation)-1):
        clean_questions.append(utilities.cleanText(memoryLine[conversation[i]]))
        clean_answers.append(utilities.cleanText(memoryLine[conversation[i+1]]))

min_line_length = 2
max_line_length = 20
clean_questions_short = []
clean_answers_short = []
for i, element in enumerate(clean_questions):
    if len(element.split()) > min_line_length and len(element.split()) <= max_line_length:
        ans = "<SOS> " + clean_answers[i] + " <EOS>"
        clean_answers_short.append(ans)
        clean_questions_short.append(element)

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer.fit_on_texts(clean_questions_short + clean_answers_short)
input_sequences = tokenizer.texts_to_sequences(clean_questions_short)
output_sequences = tokenizer.texts_to_sequences(clean_answers_short)
word2idx_vocab = tokenizer.word_index
max_len_input = max(len(s) for s in input_sequences)

threshold = 20
lowWordCountList = []
word2idx_vocab[len(word2idx_vocab)] = "<UNK>"
for k, v in tokenizer.word_counts:
    if v <= threshold:
        lowWordCountList.append(k)
