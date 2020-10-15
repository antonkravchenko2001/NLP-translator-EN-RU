from flask import Flask, render_template, request
from Model import encode, translate
app = Flask(__name__)


@app.route("/translator", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        print(request.form)
        translation = translate(encode(request.form["sentence"]))
        return render_template("translator.html", translation=translation)
    return render_template("translator.html")


if __name__ == "__main__":
    app.run(debug=True)







