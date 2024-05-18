import streamlit as st
import unicodedata
import re
import string
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nlp_bucket1 import Encoder, Decoder, evaluate_sentence, predict_greedy, predict_beam

# load tokenizers
tokenizer_urdu = None
with open('./tokenizer_urdu.pkl', 'rb') as f:
    tokenizer_urdu = pickle.load(f)

tokenizer_roman = None
with open('./tokenizer_roman.pkl', 'rb') as f:
    tokenizer_roman = pickle.load(f)

vocab_inp_size = len(tokenizer_urdu.word_index) + 1
vocab_tar_size = len(tokenizer_roman.word_index) + 1
embedding_dim = 256
units = 1024
steps_per_epoch = 17300
batch_size = 64
num_layers = 1


# load and instantiate encoder / decoder
encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_size, num_layers)
decoder = Decoder(vocab_tar_size, embedding_dim, units*2, batch_size, num_layers)

dummy_input = tf.constant([[0]])
dummy_hidden = encoder.build_initial_states(1)
encoder(dummy_input, dummy_hidden)
decoder_hidden = [tf.zeros((1, units * 2)) for _ in range(num_layers * 2)]
decoder(dummy_input, decoder_hidden, tf.random.uniform((1, 10, units * 2), dtype=tf.float32))


print('abt to request weights')

encoder_weights_file = 'encoder_weights.h5'
decoder_weights_file = 'decoder_weights.h5'
encoder_weights_url = 'https://www.dropbox.com/scl/fi/aga3ldlbb7n3qio2rjk6n/encoder_weights.h5?rlkey=2zl5uyuawpllr5pk8e87cduk9&st=ksbp0ibf&dl=1'
decoder_weights_url = 'https://www.dropbox.com/scl/fi/tgc56zpo6vf0b38eznd3t/decoder_weights.h5?rlkey=8lb298do6q5kykmty63rjd5ua&st=c4aw1pzf&dl=1'

urllib.request.urlretrieve(encoder_weights_url, encoder_weights_file)
urllib.request.urlretrieve(decoder_weights_url, decoder_weights_file)

encoder.load_weights(encoder_weights_file)
decoder.load_weights(decoder_weights_file)


print('got weights')

urdu_alphabet = [
    '\u0627', '\u0622', '\u0628', '\u067E', '\u062A', '\u0679', '\u062B',
    '\u062C', '\u0686', '\u062D', '\u062E', '\u062F', '\u0688', '\u0630',
    '\u0631', '\u0691', '\u0632', '\u0698', '\u0633', '\u0634', '\u0635',
    '\u0636', '\u0637', '\u0638', '\u0639', '\u063A', '\u0641', '\u0642',
    '\u06A9', '\u06AF', '\u0644', '\u0645', '\u0646', '\u06BA', '\u0648',
    '\u06C1', '\u06BE', '\u0621', '\u06CC', '\u0626', '\u0624', '\u0649',
    '\u06D2', '\u0651', '\u0670'
]

urdu_digits = [
    '\u0660', '\u0661', '\u0662', '\u0663', '\u0664',
    '\u0665', '\u0666', '\u0667', '\u0668', '\u0669'
]


def preprocess(inp):
    # following the same steps used to train the model

    def unicode_to_ascii(s):
      return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    
    in_ascii = unicode_to_ascii(inp)
    # white space b/w punctuation marks
    white_space_removed = re.sub(f'([{string.punctuation}])', r" \1 ", in_ascii)
    white_space_removed = re.sub(r'[" "]+', " ", white_space_removed)
    # remove alphanumeric
    alphanumeric_removed = re.sub(r"[^{}{}{}a-zA-Z0-9]+".format(''.join(urdu_alphabet), ''.join(urdu_digits), re.escape(string.punctuation)), " ", white_space_removed)
    # remove trailing / leading white space
    extra_spaces_removed = alphanumeric_removed.strip()
    # add eos and sos tokens
    return '<start> ' + extra_spaces_removed + ' <end>'


# def tokenize(inp):
#     # tokenize input
#     tokenized = tokenizer_urdu.texts_to_sequences(inp)
#     # add padding
#     tokenized_and_padded = pad_sequences(tokenized, padding='post')
#     return tokenized_and_padded


st.title("Urdu to Roman Urdu Transliterator")
st.caption("I'm learning how to deploy models and do model inference on the web using Streamlit.")
# take user input
user_input = st.text_input("Please enter a sentence in Urdu:")
button_clicked = st.button("Translate")

output = None

if button_clicked:
    # preprocess user input
    preprocessed_input = preprocess(user_input)
    # # tokenize user input
    # tokenized_input = tokenize(preprocessed_input)
    # pass input to model
    output = evaluate_sentence(preprocessed_input, encoder, decoder, num_layers)
    output = output[len('<start> '):]
    output = output[:-len(' <end>')]
# pass it to the model
# print the result
st.write("Output:")
st.write(output)

