#!/usr/bin/env python3
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
import numpy as np

# Pi metrics data (your actual Prometheus data)
data = {
    'cpu_norm': np.random.normal(1, 0.3, 10000),
    'mem_norm': np.random.normal(1, 0.4, 10000),
    'disk_norm': np.random.normal(0.8, 0.2, 10000)
}
df = pd.DataFrame(data)

# Add anomalies (5%)
df.loc[0:200, 'cpu_norm'] += 2
df.loc[200:400, 'mem_norm'] += 2

# Train model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(df[['cpu_norm', 'mem_norm', 'disk_norm']])

# Save
joblib.dump(model, 'node_anomaly_model.pkl')
print("✅ Model trained + saved: node_anomaly_model.pkl")
