#!/bin/bash
# Keep the application as systemd's main PID; tee mirrors output to the journal.
set -euo pipefail
log_file=$1
shift
mkdir -p "$(dirname "$log_file")"
touch "$log_file"
exec "$@" > >(/usr/bin/tee -a "$log_file") 2>&1
