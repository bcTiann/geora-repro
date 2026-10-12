"""External process watchdog for the existing GPU comparison driver.

Pass the same arguments as prove_candidates_gpu.py. The requested limit is
the whole launch, at most1200 seconds. The child gets a shorter internal
deadline; TERM and KILL target its process group before the outer limit.
No scheduler job is submitted by this wrapper.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
DRIVER = ROOT / "prove_candidates_gpu.py"


def main():
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        return subprocess.run([sys.executable,str(DRIVER),"--help"]).returncode
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime-limit-seconds",type=float,default=1200)
    parser.add_argument("--output-dir",type=Path,required=True)
    args, remaining = parser.parse_known_args()
    if not 30<args.runtime_limit_seconds<=1200:
        parser.error("The total wall-clock limit must be >30 and <=1200 seconds")
    started = time.monotonic()
    deadline = started+args.runtime_limit_seconds
    run_id = str(uuid.uuid4())
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", GEORA_SPEED_WATCHDOG_ID=run_id)
    child = subprocess.Popen([sys.executable,str(DRIVER),*remaining,
        "--output-dir",str(args.output_dir),"--runtime-limit-seconds",str(args.runtime_limit_seconds-15)],
        env=environment,start_new_session=True)
    timed_out = False
    try:
        code = child.wait(timeout=max(0.001,deadline-time.monotonic()-8))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        code = 124
        # Only touch a report created by this child, never an existing input.
        report_path = args.output_dir / "comparison.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report.get("watchdog_run_id")==run_id and report.get("diagnostic_only") is True:
                report.update(status="timed_out",error="External GPU-driver wall-clock watchdog terminated the process group")
                temporary = report_path.with_suffix(".json.watchdog.tmp")
                temporary.write_text(json.dumps(report,indent=2)+"\n")
                temporary.replace(report_path)
    print(json.dumps(dict(watchdog_status="timed_out" if timed_out else "child_finished",
        child_exit_code=code, elapsed_seconds=time.monotonic()-started,
        total_wall_clock_limit_seconds=args.runtime_limit_seconds,
        real_training_optimizer_steps_added=0)),flush=True)
    return code


if __name__=="__main__":
    sys.exit(main())
