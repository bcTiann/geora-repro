"""Check inference/training imports without constructing a model or GPU client."""

from datetime import datetime, timezone
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import sys


def main():
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": False,
        "gpu_tested": False,
        "objects_instantiated": False,
        "imports": [],
    }
    imports = [
        ("vllm", ("LLM", "SamplingParams")),
        ("trl", ("GRPOTrainer", "GRPOConfig")),
        ("trl.generation.vllm_client", ("VLLMClient",)),
    ]
    try:
        for name, attributes in imports:
            report["stage"] = name
            module = importlib.import_module(name)
            for attribute in attributes:
                getattr(module, attribute)
            report["imports"].append({"module": name, "attributes": list(attributes)})
        report["versions"] = {
            name: version(name) for name in ("vllm", "torch", "trl", "transformers", "peft")
        }
        report["stage"] = "complete"
        report["passed"] = True
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)
    output = Path(__file__).parent / "evidence" / "imports.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
