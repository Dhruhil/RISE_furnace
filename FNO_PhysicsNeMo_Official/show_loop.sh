#!/bin/bash
# Show the exact training loop from train.py (lines around "for epoch" and "train_loader")
echo "=== Lines around training loop ==="
grep -n "for epoch\|for x, y\|train_loader\|loss_data\|loss =\|lam\|F.mse_loss\|loss.backward\|get_lambda\|fno_physics\|model.train\|total_loss" train.py | head -40
echo ""
echo "=== Full training loop block (line numbers) ==="
awk '/for epoch in range/,/elapsed = time/' train.py | head -60
