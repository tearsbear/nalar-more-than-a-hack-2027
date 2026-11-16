# Reproduce NALAR

Use Python 3.12.4 from the repository root. The reference run used macOS arm64.
Package installation needs network access. Generation runs offline on a CPU.

```bash
python3.12 -m venv .venv-nalar
source .venv-nalar/bin/activate
python -m pip install -r requirements-nalar.txt
python -m pip check
python -m unittest discover -s tests -q
```

## Supply the match data

Unpack the organizer bundle separately and point `MTAH_DATA_DIR` at the directory
containing `tactical/`. Using the repository's ignored `data/` directory also works.
NALAR reads the tactical tracking metadata XML, tracking TXT and Smart Tagging
XML. [DATA.md](DATA.md) describes the bundle and [games.py](games.py) defines its
paths. Neither video nor ground truth is required.

```bash
export MTAH_DATA_DIR="/absolute/path/to/competition-data"
```

## Generate and verify

```bash
python -m nalar generate \
  --data-dir "$MTAH_DATA_DIR" \
  --config configs/nalar_v1.json \
  --output artifacts/submission/NALAR.csv \
  --manifest artifacts/submission/run.json
python evaluation/validate_submission.py artifacts/submission/NALAR.csv
cp TECHNICAL_REPORT.md artifacts/submission/NALAR.md
python - <<'PY'
import hashlib
from pathlib import Path

csv = Path("artifacts/submission/NALAR.csv")
expected = "ca86ec5deb38cc113277fbf9dca25f5056839575bcd4fafc81cbd155018ecfae"
actual = hashlib.sha256(csv.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"CSV hash mismatch: {actual}")
print(actual)
PY
```

The validator checks the format of the 199,319-row CSV. The hash verifies that
its bytes match the submission. The local manifest records the source, inputs,
configuration and output fingerprints, plus the environment and runtime. The
manifest can differ across runs.

For diagnostics without labels, run
`python -m nalar diagnose --data-dir "$MTAH_DATA_DIR" --config configs/nalar_v1.json`.
It reports aggregate changes after removing tags, dead-ball evidence or
interpolation, or adding seeded dropout or distance noise.

## Submit

After publication, repeat these steps from a fresh clone of the public release
commit. The designated team member submits one response as **NALAR** through the
organizer's emailed form, with `NALAR.csv`, `NALAR.md` and the public repository
URL. The deadline is 17 November 2026 at 23:59 CET (18 November, 05:59 WIB).
See [SUBMISSION.md](SUBMISSION.md) for the official requirements.

Keep the CSV, manifests, data bundle and any match-derived debug files local.
The organizer's public GT is for pipeline sanity checks only. NALAR generation
and its synthetic tests do not read it.
