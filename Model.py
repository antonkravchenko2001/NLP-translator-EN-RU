import tensorflow as tf
import json
import numpy as np


with open('C:\\Users\\anton\\PycharmProjects\\Machine_translation\\input_dict.json', 'r') as fp:
    input_dict = json.load(fp)
with open('C:\\Users\\anton\\PycharmProjects\\Machine_translation\\target_dict_r.json', 'r') as fp:
    target_dict_r = json.load(fp)
encoder_model = tf.keras.models.load_model('C:\\Users\\anton\\PycharmProjects\\Machine_translation\\encoder.h5')
decoder_model = tf.keras.models.load_model('C:\\Users\\anton\\PycharmProjects\\Machine_translation\\decoder.h5')


def encode(seq):
    seq = seq.lower()
    seq = seq.split()
    for i in range(len(seq)):
        seq[i] = input_dict[seq[i]]
    seq = np.array(seq)
    seq = np.reshape(seq, (1, -1))
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=8, padding="pre")
    return seq


def translate(input_seq):
    h1, c1 = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = 1
    eos = 2
    output_sentence = []
    current_state = [h1, c1, h1, c1]
    for _ in range(20):
        output_tokens, h3, c3, h4, c4 = decoder_model.predict([target_seq] + current_state)
        idx = np.argmax(output_tokens[0, 0, :])
        if eos == idx:
            break
        word = ''
        if idx > 0:
            word = target_dict_r[str(idx)]
            output_sentence.append(word)
        target_seq[0, 0] = idx
        current_state = [h3, c4, h4, c4]
    return ' '.join(output_sentence)

