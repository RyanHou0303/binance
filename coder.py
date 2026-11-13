import json
import subprocess
from pathlib import Path
def write_signal() -> Path:
    base = Path(__file__).resolve().parent
    plan = json.loads((base / "plan.json").read_text(encoding="utf-8"))
    output = base / "generated_signal.py"

    prompt = f"""
You are a Python quantitative developer.
Implement the strategy described in the following plan.

Requirements:
- Define a class named Signal.
- Implement get_weight(self, close: pd.DataFrame) -> pd.DataFrame.
- Input rows are timestamps; columns are assets.
- Return weights with exactly the same index and columns.
- Positive weights are long; negative weights are short.
- All weights must be finite; absolute weights per row must sum to at most 1.
- Weights execute at the current bar's open, so use only earlier bars.
- Use zero weights during warmup or when required observations are missing.
- Only closing prices are available. Raise an error if other inputs are required.
- Follow the plan; document any necessary assumptions in comments.
- Only use pandas, numpy, and the Python standard library.
- Return only Python source code, without Markdown fences.
- Do not use tools, modify files, or run the code.

Plan:
{json.dumps(plan, indent=2, ensure_ascii=False)}
"""
    result = subprocess.run(
        [
            "codex", "exec",
            "--sandbox", "read-only",
            "--output-last-message", str(output),
            "-",
        ],
        input=prompt,
        text=True,
        capture_output=True,
        cwd=base,
    )
    return output

if __name__ == "__main__":
    path = write_signal()
    print(f"Saved signal code to: {path}")
