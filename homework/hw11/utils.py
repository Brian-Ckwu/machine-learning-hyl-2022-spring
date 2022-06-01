import json
from pathlib import Path

def load_json(file: str):
    return json.loads(Path(file).read_bytes())