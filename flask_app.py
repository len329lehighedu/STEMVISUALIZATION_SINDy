from flask import Flask, render_template
from bokeh.embed import server_document

app = Flask(__name__)

# Link to Bokeh Server 
BOKEH_URL = "http://localhost:5006/main" 

@app.route("/")
def storyline():
    script = server_document(BOKEH_URL)
    return render_template("storyline.html", script=script) 

@app.route("/about")
def about():
    script = server_document(BOKEH_URL)
    return render_template("about.html", script=script) 

@app.route("/instructions")
def instructions():
    script = server_document(BOKEH_URL)
    return render_template("instructions.html", script=script) 

@app.route("/questions")
def questions():
    script = server_document(BOKEH_URL)
    return render_template("questions.html", script=script) #


if __name__ == "__main__":
    app.run(port=8080, debug=True)