#!/bin/bash
LOG="$1"
PATIENCE="${2:-12}"
PID="${3}"
MIN_DELTA=0.00001

echo "[monitor] log=$LOG patience=$PATIENCE pid=$PID"

while true; do
    # 提取最新 epoch 和 val loss
    LATEST=$(grep -oP 'Epoch\s+\d+/\d+\s+\|\s+train:\s+[\d.]+\s+val:\s+[\d.]+' "$LOG" 2>/dev/null | tail -20)
    
    if [ -z "$LATEST" ]; then
        echo "[monitor] $(date +%H:%M:%S) waiting for epochs..."
        sleep 60
        continue
    fi
    
    CURRENT_EPOCH=$(echo "$LATEST" | tail -1 | grep -oP 'Epoch\s+\K\d+')
    
    # 找到最佳 val loss 和对应的 epoch
    BEST_VAL=999
    BEST_EPOCH=0
    while IFS= read -r line; do
        EPOCH=$(echo "$line" | grep -oP 'Epoch\s+\K\d+')
        VAL=$(echo "$line" | grep -oP 'val:\s+\K[\d.]+')
        if [ -n "$VAL" ] && (( $(echo "$VAL < $BEST_VAL" | bc -l 2>/dev/null || echo 0) )); then
            BEST_VAL="$VAL"
            BEST_EPOCH="$EPOCH"
        fi
    done <<< "$LATEST"
    
    EPOCHS_SINCE_BEST=$((CURRENT_EPOCH - BEST_EPOCH))
    
    # 最新5个 epoch
    RECENT=$(echo "$LATEST" | tail -5 | grep -oP 'E\d+/' | tr '\n' ' ' || echo "N/A")
    
    echo "[monitor] $(date +%H:%M:%S) Epoch $CURRENT_EPOCH | best: $BEST_VAL (E$BEST_EPOCH) | stalled: $EPOCHS_SINCE_BEST epochs"
    
    if [ "$EPOCHS_SINCE_BEST" -ge "$PATIENCE" ] 2>/dev/null; then
        echo "[monitor] $(date +%H:%M:%S) EARLY STOP! val loss no improvement for $EPOCHS_SINCE_BEST ≥ $PATIENCE epochs"
        if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
            kill -TERM "$PID"
            echo "[monitor] Sent SIGTERM to PID $PID"
            sleep 5
            if kill -0 "$PID" 2>/dev/null; then
                kill -9 "$PID"
                echo "[monitor] Sent SIGKILL to PID $PID"
            fi
        fi
        exit 0
    fi
    
    if [ "$CURRENT_EPOCH" -ge 100 ] 2>/dev/null; then
        echo "[monitor] $(date +%H:%M:%S) Training finished (100 epochs)."
        exit 0
    fi
    
    sleep 60
done
