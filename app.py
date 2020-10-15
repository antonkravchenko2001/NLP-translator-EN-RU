from flask import Flask, render_template, request
from Model import encode, translate, input_dict
app = Flask(__name__)


@app.route("/translator", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        for word in request.form["sentence"].split():
            if word not in input_dict.keys():
                return render_template("translator.html", translation='No such word in the dictionary')
        if request.form["sentence"] == "":
            return render_template("translator.html", translation="")
        translation = translate(encode(request.form["sentence"]))
        return render_template("translator.html", translation=translation)
    return render_template("translator.html")


if __name__ == "__main__":
    app.run()






