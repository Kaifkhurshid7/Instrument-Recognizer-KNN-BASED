import pickle
import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).parent / 'reference_database.pkl'

if not DB_PATH.exists():
    print('ERROR: database file not found at', DB_PATH)
    raise SystemExit(1)

with open(DB_PATH, 'rb') as f:
    db = pickle.load(f)

print('Database summary:')
print('  Classes:', len(db))
print()
for k, v in db.items():
    fps = v.get('fingerprints', [])
    avg = np.array(v.get('average_vector'))
    print(f"  - {k}: fingerprints={len(fps)}, average_vector_shape={avg.shape}")
