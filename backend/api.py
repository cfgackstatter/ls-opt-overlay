from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.models import SimulationParams, SimulationResult, MonteCarloParams, MonteCarloResult
from backend.simulator import run_simulation
from backend.monte_carlo import run_monte_carlo

app = FastAPI(title="LS Opt Overlay API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/simulate")
def simulate(params: SimulationParams) -> SimulationResult:
    return run_simulation(params)

@app.post("/monte_carlo")
def monte_carlo(params: MonteCarloParams) -> MonteCarloResult:
    return run_monte_carlo(params)
