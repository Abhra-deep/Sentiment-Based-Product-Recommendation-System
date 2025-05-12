from flask import Flask, render_template, request
from model import get_sentiment_recommendations, dataframe_to_html

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    user = request.form.get('user')
    recommendation_resp = get_sentiment_recommendations(user)

    # Check if the response is a string (indicating user does not exist)
    if isinstance(recommendation_resp, str):
        return render_template('index.html', user=user)

    # If the response is a DataFrame, convert it to HTML for rendering
    recommendation_html = dataframe_to_html(recommendation_resp)
    return render_template('index.html', user=user, recommendation=recommendation_html)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
