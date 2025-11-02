from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api import api_football_controller

app = FastAPI(
    title="Anabet API",
    version="2.0.0",
    description="API d'analyse sportive avec IA et intégration API-Football"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(api_football_controller.router)

@app.get("/")
async def root():
    return {
        "message": "Anabet API - Système d'analyse sportive avec IA",
        "version": "2.0.0",
        "features": [
            "Prédictions IA avec Ollama",
            "Intégration API-Football",
            "Système d'abonnement freemium",
            "Cache et rate limiting"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)