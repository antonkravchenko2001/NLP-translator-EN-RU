from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, SpatialDropout1D,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split



num_samples = 40000
max_num_words = 10000
latent_dim = 200

"""
data parser
"""


def parse(data_path):
    inputs = []
    targets = []
    c = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[: len(lines) - 1]:
        c += 1
        if c > 40000:
            break
        input_, target, _ = line.split('\t')
        target = 'start ' + target + " end"
        inputs.append(input_)
        targets.append(target)
    return inputs, targets


def tokenize(text):
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(text)
    text = tokenizer.texts_to_sequences(text)
    return text, tokenizer.word_index, {v: k for k, v in tokenizer.word_index.items()}


"""
function returning max sequence length of all samples
"""


def max_sequense_length(text):
    length_list = []
    for sentence in text:
        l = 0
        for _ in sentence:
            l += 1
        length_list.append(l)
    return max(length_list)


def batch_generator(x, y, batch_size=128):
    while True:
        for i in range(0, len(x), batch_size):
            encoder_inputs = pad_sequences(x[i:i+batch_size], maxlen=max_sequense_length(input_text), padding='pre')
            decoder_inputs = np.zeros((batch_size, max_sequense_length(target_text)-1))
            decoder_outputs = np.zeros((batch_size, max_sequense_length(target_text)-1, len(target_dict)+1))
            for j, txt in enumerate(y[i:i+batch_size]):
                for t, char in enumerate(txt):
                    if t > 0:
                        decoder_outputs[j, t-1, char] = 1
                    if t < max_sequense_length(target_text)-1:
                        decoder_inputs[j, t] = char
                if t < max_sequense_length(target_text)-1:
                    decoder_outputs[j, t:, 0] = 1
            yield ([encoder_inputs, decoder_inputs], decoder_outputs)




def val_data(x, y):
    encoder_inputs = pad_sequences(x, maxlen=max_sequense_length(input_text), padding='pre')
    decoder_inputs = np.zeros((len(y), max_sequense_length(target_text)-1))
    decoder_outputs = np.zeros((len(y), max_sequense_length(target_text)-1, len(target_dict)+1))
    for i, txt in enumerate(y):
        for t, char in enumerate(txt):
            if t > 0:
                decoder_outputs[i, t-1, char] = 1
            if t < max_sequense_length(target_text)-1:
                decoder_inputs[i, t] = char
        if t < max_sequense_length(target_text)-1:
            decoder_outputs[i, t:, 0] = 1
    return ([encoder_inputs, decoder_inputs], decoder_outputs)


"""
load and trim/split data
"""


input_text, target_text = parse('C:\\Users\\anton\\PycharmProjects\\Machine_translation\\rus-eng\\rus.txt')
input_text, input_dict, input_dict_r = tokenize(input_text)
target_text, target_dict, target_dict_r = tokenize(target_text)
input_text_train, input_text_test, target_text_train, target_text_test = train_test_split(input_text, target_text, train_size=0.9)

"""
create test and train batch generators
"""
train_generator = batch_generator(input_text_train, target_text_train, batch_size=100)
test_generator = batch_generator(input_text_test, target_text_test, batch_size=100)


if __name__ == "__main__":
    """
    instantiate model
    """
    enc_inp = Input(shape=(None,))
    enc_emb_layer = Embedding(len(input_dict)+1, latent_dim)
    enc_emb = enc_emb_layer(enc_inp)
    enc_dropout = SpatialDropout1D(0.3)
    enc_emb = enc_dropout(enc_emb)
    enc_lstm_layer = LSTM(latent_dim, return_state=True)
    enc_out, state_h, state_c =  enc_lstm_layer(enc_emb)
    enc_states = [state_h, state_c]
    dec_inp = Input(shape=(None,))
    dec_emb_layer = Embedding(len(target_dict)+1, latent_dim)
    dec_emb = dec_emb_layer(dec_inp)
    dec_dropout = SpatialDropout1D(0.2)
    dec_emb = dec_dropout(dec_emb)
    dec_lstm_layer = LSTM(latent_dim, return_sequences=True, return_state=True)
    dec_out, _, _ = dec_lstm_layer(dec_emb, initial_state=enc_states)
    dec_out_droput = Dropout(0.3)
    dec_out = dec_out_droput(dec_out)
    dec_dense_layer = Dense(len(target_dict)+1, activation="softmax")
    dec_out = dec_dense_layer(dec_out)
    model = Model([enc_inp, dec_inp], dec_out)

    """
    train model
    """
    opt = Adam(0.03)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["acc"])
    r = model.fit(train_generator, shuffle=True, epochs=20, steps_per_epoch=200)
    model.save('C:\\Users\\anton\\PycharmProjects\\Machine_translation\\training_model.h5')
    """
    creating prediction model
    """
    encoder_model = Model(enc_inp, enc_states)
    encoder_model.save('C:\\Users\\anton\\PycharmProjects\\Machine_translation\\encoder_model.h5')
    dec_input_h = Input(shape=(latent_dim,))
    dec_input_c = Input(shape=(latent_dim,))
    dec_input_states = [dec_input_h, dec_input_c]
    dec_inputs_single = Input(shape=(1,))
    dec_emb_single = dec_emb_layer(dec_inputs_single)
    dec_outputs_single, h, c = dec_lstm_layer(dec_emb_single, initial_state=dec_input_states)
    dec_states = [h, c]
    dec_outputs = dec_dense_layer(dec_outputs_single)
    decoder_model = Model([dec_inputs_single] + dec_input_states, [dec_outputs] + dec_states)
    decoder_model.save('C:\\Users\\anton\\PycharmProjects\\Machine_translation\\decoder_model.h5')
