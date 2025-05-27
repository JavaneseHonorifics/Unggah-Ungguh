import json
import os

def _read_json(file_json_path: str):
    with open(file_json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(results, output_dir: str):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=2, ensure_ascii=False)