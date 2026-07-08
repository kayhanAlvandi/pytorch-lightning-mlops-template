import os
import sqlite3

DB = "mlflow.db"
print("exists:", os.path.exists(DB), "size:", os.path.getsize(DB) if os.path.exists(DB) else 0)

c = sqlite3.connect(DB)
cur = c.cursor()

print("\nEXPERIMENTS:")
for r in cur.execute("SELECT experiment_id, name, lifecycle_stage FROM experiments"):
    print("  ", r)

print("\nRUN COUNT:", cur.execute("SELECT count(*) FROM runs").fetchone()[0])
print("RUNS (first 20):")
for r in cur.execute("SELECT run_uuid, name, status, experiment_id FROM runs LIMIT 20"):
    print("  ", r)

print("\nREGISTERED MODELS:")
for r in cur.execute("SELECT name FROM registered_models"):
    print("  ", r)

print("\nMODEL VERSIONS:")
for r in cur.execute("SELECT name, version, run_id, current_stage FROM model_versions LIMIT 20"):
    print("  ", r)

c.close()
