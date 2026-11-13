import subprocess
import json
from pathlib import Path

def think(thesis: str) -> str:
    prompt = f"""
You are a quantitative research assistant.
Turn the user's thesis into a concrete, backtestable strategy.

Constraints:
- Input: daily closing prices, indexed by time, with one column per asset.
- Generate target weights for execution at the next bar's open.
- Positive weights are long; negative weights are short.
- The sum of absolute weights must not exceed 1.
- Avoid look-ahead bias.

Respond in English with:
1. Hypothesis and rationale
2. Factor formula
3. Portfolio weighting rules
4. Validation plan

State any assumptions you introduce.
Do not invent backtest results.
Return only the plan, without using tools or modifying files.

Thesis:
{thesis}
"""

    result = subprocess.run(
        [
            "codex", "exec",
            "--sandbox", "read-only",
            "--output-schema", "plan.schema.json",
            "--output-last-message", "plan.json",
            "-",
        ],
        input=prompt,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    path = Path("plan.json")
    plan = json.loads(path.read_text(encoding="utf-8"))

    path.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return plan


if __name__ == "__main__":
    thesis = input("Enter your thesis: ")
    plan = think(thesis)
    print(plan)