from flask import Flask, render_template, request, abort
from ML.Model import encode, translate, input_dict
app = Flask(__name__)


urls = ['translate']


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        for word in request.form["sentence"].split():
            if word.lower() not in input_dict.keys():
                return render_template("translator.html", translation=f'No such word in the dictionary: {word}')
        if request.form["sentence"] == "":
            return render_template("translator.html", translation="")
        translation = translate(encode(request.form["sentence"]))
        return render_template("translator.html", translation=translation)
    return render_template("translator.html")

@app.errorhandler(404)
def page_not_found(error):
   return render_template('404.html', title = '404'), 404



if __name__ == "__main__":
    app.run(debug=True)
