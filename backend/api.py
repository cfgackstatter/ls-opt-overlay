from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
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

@app.get("/defaults", response_model=SimulationParams)
def get_defaults() -> SimulationParams:
    """Return default simulation parameters. Single source of truth is models.py."""
    return SimulationParams()

@app.post("/simulate")
async def simulate(params: SimulationParams) -> SimulationResult:
    return await run_in_threadpool(run_simulation, params)

@app.post("/monte_carlo")
async def monte_carlo(params: MonteCarloParams) -> MonteCarloResult:
    # Process pool inside; run the driver off the event loop so /health stays up
    return await run_in_threadpool(run_monte_carlo, params)
