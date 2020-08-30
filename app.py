from flask import Flask, redirect, url_for, render_template, request, session, flash
from Model import input_dict, target_dict, target_dict_r, max_sequense_length, target_text, input_dict_r, input_text
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)


encoder_model = tf.keras.models.load_model("encoder_model.h5")
decoder_model = tf.keras.models.load_model("decoder_model.h5")

"""
encode sentence before translation
"""


def encode(seq):
    seq = seq.lower()
    seq = seq.split()
    for i in range(len(seq)):
        seq[i] = input_dict[seq[i]]
    seq = np.array(seq)
    seq = np.reshape(seq, (1, -1))
    seq = pad_sequences(seq, maxlen=max_sequense_length(input_text), padding="pre")
    return seq


"""
translate
"""


def translate(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_dict['start']
    eos = target_dict['end']
    output_sentence = []
    for _ in range(max_sequense_length(target_text)):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])
        if eos == idx:
            break
        word = ''
        if idx > 0:
            word = target_dict_r[idx]
            output_sentence.append(word)
        target_seq[0, 0] = idx
        states_value = [h, c]
    return ' '.join(output_sentence)


@app.route("/translator", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        print(request.form)
        translation = translate(encode(request.form["sentence"]))
        return render_template("translator.html", translation=translation)
    return render_template("translator.html")


if __name__ == "__main__":
    app.run(debug=True)







