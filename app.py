from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from prometheus_client import make_asgi_app
import uvicorn

app = FastAPI(title="Pi Anomaly Detector")
model = joblib.load("node_anomaly_model.pkl")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

class NodeMetrics(BaseModel):
    cpu_norm: float = 0.0
    mem_norm: float = 0.0

@app.post("/score")
async def score(m: NodeMetrics):
    X = np.array([[m.cpu_norm, m.mem_norm]])
    pred = int(model.predict(X)[0])  # Convert numpy bool to int
    score = float(-model.decision_function(X)[0])
    return {
        "is_anomaly": pred == -1,
        "anomaly_score": score,
        "confidence": min(score, 1.0)
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
