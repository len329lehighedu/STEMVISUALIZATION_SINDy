from flask import Flask, render_template
from bokeh.embed import server_document

app = Flask(__name__)

BOKEH_URL = "http://localhost:5006/main"


@app.route("/")
def index():
    script = server_document(BOKEH_URL)
    return render_template(
        "index.html",
        script=script
    )


@app.route("/fragment/storyline")
def storyline():
    return render_template("fragment/storyline.html")


@app.route("/fragment/about")
def about():
    return render_template("fragment/about.html")


@app.route("/fragment/training")
def train():
    return render_template("fragment/training.html")


@app.route("/fragment/testing")
def test():
    return render_template("fragment/testing.html")


@app.route("/fragment/questions")
def questions():
    return render_template("fragment/questions.html")


if __name__ == "__main__":
    app.run(port=8080, debug=True)