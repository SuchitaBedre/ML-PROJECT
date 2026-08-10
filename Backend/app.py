# =====================================================
# IMPORT LIBRARIES
# =====================================================

from fastapi import FastAPI

from Backend.routes.search import router as search_router
from Backend.routes.recommendation import router as recommendation_router
from Backend.routes.prediction import router as prediction_router
from Backend.routes.nutrition import router as nutrition_router
from Backend.routes.assistant import router as assistant_router
from Backend.routes.dashboard import router as dashboard_router
from Backend.routes import health_score
from Backend.routes import sentiment
from Backend.routes import review
from Backend.routes.recipes import router as recipes_router
from Backend.routes import nutrition
# =====================================================
# CREATE FASTAPI APP
# =====================================================

app = FastAPI(
    title="AI Powered Recipe Recommendation API",
    description="Backend API for Recipe Search, Recommendation, Rating Prediction and Nutrition",
    version="1.0.0"
)


# =====================================================
# INCLUDE ROUTERS
# =====================================================

app.include_router(
    search_router,
    prefix="/search",
    tags=["Recipe Search"]
)

app.include_router(
    recommendation_router,
    prefix="/recommendation",
    tags=["Recommendation"]
)

app.include_router(
    prediction_router,
    prefix="/prediction",
    tags=["Prediction"]
)

app.include_router(
    nutrition_router,
    prefix="/nutrition",
    tags=["Nutrition"]
)

app.include_router(
    assistant_router
)

app.include_router(
    dashboard_router,
    prefix="/dashboard",
    tags=["Dashboard"]
)

app.include_router(
    health_score.router
)

app.include_router(

    sentiment.router

)

app.include_router(
    review.router
)

app.include_router(
    recipes_router
)
app.include_router(
    nutrition.router
)

# =====================================================
# HOME API
# =====================================================

@app.get("/")
def home():
    return {
        "status": "success",
        "message": "AI Powered Recipe Recommendation Backend Running"
    }

print("\n========== REGISTERED ROUTES ==========")

for route in app.routes:
    try:
        print(route.path, route.methods)
    except Exception:
        pass

print("=======================================\n")