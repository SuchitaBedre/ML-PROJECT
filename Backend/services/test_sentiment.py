from Backend.services.sentiment_service import predict_sentiment


reviews = [

    "This recipe is amazing. My family loved it.",

    "The recipe was terrible and tasteless.",

    "The recipe is okay, nothing special."

]


for review in reviews:

    result = predict_sentiment(review)

    print("\nReview:")
    print(review)

    print("Result:")
    print(result)