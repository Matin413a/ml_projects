from transformers import pipeline


classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)


text = "this movie was really bad! "

result = classifier(text)

print(result)