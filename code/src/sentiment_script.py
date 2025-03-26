from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    sentiment = "Positive" if compound > 0.05 else "Negative" if compound < -0.05 else "Neutral"
    return compound

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = sys.argv[1]
        score = get_sentiment_score(input_text)
        print(score)  # <-- print this so the shell can capture it
    else:
        print("No input text provided.")
