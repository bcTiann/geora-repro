#!/bin/bash
# Stream a submitted job's log until it leaves the Slurm queue.
# Ctrl+C stops viewing only; it does not cancel the GPU job.

set -euo pipefail
if [[ $# -ne 2 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: /bin/bash jobs/watch_gpu_check.sh JOBID LOG_PATH" >&2
  exit 2
fi
geora_job_id="$1"
geora_log="$2"
geora_tail_pid=""
cleanup() {
  if [[ -n "$geora_tail_pid" ]]; then
    kill "$geora_tail_pid" 2>/dev/null || true
    wait "$geora_tail_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# GNU tail on Setonix follows creation and replacement of the Slurm output file.
echo "Live log: $geora_log"
echo "Ctrl+C stops viewing; cancel the job separately with: scancel $geora_job_id"
tail --sleep-interval=1 --max-unchanged-stats=1 -n +1 -F -- "$geora_log" &
geora_tail_pid="$!"
geora_previous_state=""
while true; do
  geora_state=$(squeue --noheader --jobs="$geora_job_id" --format='%T')
  if [[ -z "$geora_state" ]]; then
    # Allow the follower to flush output written immediately before job completion.
    sleep 2
    break
  fi
  if [[ "$geora_state" != "$geora_previous_state" ]]; then
    echo "Slurm job $geora_job_id: $geora_state"
    geora_previous_state="$geora_state"
  fi
  sleep 5
done
echo "Job $geora_job_id has left the queue. Check the log/report for pass or failure."
