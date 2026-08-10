from services.prediction_service import predict_rating

try:
    result = predict_rating("Chicken Biryani")   # Replace with a recipe that exists
    print(result)
except Exception as e:
    print("Error:", e)