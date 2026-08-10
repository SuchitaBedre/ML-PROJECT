from schemas.request import PredictionRequest

data = PredictionRequest(
    name="Pizza",
    ingredients="cheese tomato",
    tags="italian",
    description="Easy homemade pizza"
)

print(data)