#!/usr/bin/env bash
# Resilient ModelScope download for DeepSeek-V2-Lite-Chat.
# ModelScope resumes from its cache on each re-run, so retrying is safe.
cd /workspace || exit 1
for i in $(seq 1 500); do
  echo "=== attempt $i $(date -u +%H:%M:%S) ==="
  if modelscope download --model deepseek-ai/DeepSeek-V2-Lite-Chat \
        --local_dir models/DeepSeek-V2-Lite-Chat; then
    echo "=== DOWNLOAD COMPLETE at attempt $i $(date -u +%H:%M:%S) ==="
    break
  fi
  echo "=== attempt $i failed, sleep 15 ==="
  sleep 15
done
