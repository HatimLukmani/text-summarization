from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from heapq import nlargest

app = Flask(__name__)
CORS(app)

nltk.download('punkt')
nltk.download('stopwords')

def summarize_text(text, percentage=0.3):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Tokenize the text into words
    words = word_tokenize(text.lower())

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    # Calculate word frequency
    word_freq = FreqDist(words)

    # Calculate sentence scores based on word frequency
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                if len(sentence.split(' ')) < 30:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_freq[word]
                    else:
                        sentence_scores[sentence] += word_freq[word]

    # Select top sentences based on scores
    num_sentences = int(len(sentences) * percentage / 100)
    summarized_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    # Combine selected sentences to get the summary
    summary = ' '.join(summarized_sentences)
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data['text']
    percentage = float(data['percentage'])
    summary = summarize_text(text, percentage)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)