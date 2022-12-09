from flask import Flask, render_template, request
import pandas as pd

df = pd.read_csv("incd_cleaned.csv")

state_list = df["State"].to_list()
rate_list = df["Age-Adjusted Incidence Rate([rate note]) - cases per 100,000"].to_list()
rate_average = df["Age-Adjusted Incidence Rate([rate note]) - cases per 100,000"].mean()
rate_average = round(rate_average, 2)
rate_std = df["Age-Adjusted Incidence Rate([rate note]) - cases per 100,000"].std()
rate_std = round(rate_std, 2)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/info", methods=["GET"])
def analyze():
    usertext = request.args.get("usertext")
    if usertext in state_list[1:]:
        i = state_list.index(usertext)
        rate = str(rate_list[i])
    else:
        rate = "This is NOT a valid state name. Please input a valid State name!"
    return render_template("analyze.html", output = rate, usertext=usertext)

@app.route("/state/<string:name>")
def state(name):
    i = state_list.index(name)
    rate = rate_list[i]
    state = "The State name is: <u>" + name + "</u>. It has an age-adjusted incidence rate (per 100k) of <u>" + str(rate) + "</u>"
    return state
    
@app.route("/moreinfo")
def mean_std():
    info = "<center> In the United States, the average of age-adjusted incidence rate is: <b> <br>" + str(rate_average) + "</b> <br>The standard deviation is: <b><br>" + str(rate_std) + "</b> </center>"
    return info

if __name__ == "__main__":
    app.run(debug=True)
