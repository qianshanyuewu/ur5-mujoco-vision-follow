from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "assets" / "models" / "hand_landmarker.task"
URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
EXPECTED_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    temporary = TARGET.with_suffix(".task.download")
    try:
        urllib.request.urlretrieve(URL, temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != EXPECTED_SHA256:
            raise RuntimeError(f"Unexpected model checksum: {digest}")
        temporary.replace(TARGET)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded and verified {TARGET}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Failed to download hand model: {exc}", file=sys.stderr)
        raise
