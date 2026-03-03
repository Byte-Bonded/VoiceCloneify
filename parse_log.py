#!/usr/bin/env python3
"""Parse training_v3.log to understand training status."""
import re

with open('/home/pranaav/Work/projects/SEM6/VoiceCloneify/training_v3.log', 'rb') as f:
    data = f.read().decode('utf-8', errors='replace')

lines = data.replace('\r', '\n').split('\n')

# Find epoch completions and last progress
epochs_done = []
last_update = None
val_losses = []
best_models = []
errors = []

for line in lines:
    if '100%' in line and 'Epoch' in line:
        epochs_done.append(line.strip()[:120])
    m = re.search(r'Epoch (\d+).*?(\d+)/2631.*g=([0-9.]+).*d=([0-9.]+).*Dacc=([0-9.]+)', line)
    if m:
        last_update = (int(m.group(1)), int(m.group(2)), m.group(3), m.group(4), m.group(5))
    if 'Error' in line or 'Traceback' in line or 'RuntimeError' in line or 'killed' in line.lower():
        errors.append(line.strip()[:150])
    if 'Val' in line and 'loss' in line.lower():
        val_losses.append(line.strip()[:150])
    if 'best' in line.lower() and 'model' in line.lower():
        best_models.append(line.strip()[:150])
    if 'checkpoint' in line.lower() and 'saved' in line.lower():
        pass  # skip verbose checkpoint saves

print('=== EPOCHS COMPLETED ===')
for e in epochs_done[-10:]:
    print(e)

print(f'\n=== LAST UPDATE ===')
if last_update:
    print(f'Epoch {last_update[0]}, step {last_update[1]}/2631, g={last_update[2]}, d={last_update[3]}, Dacc={last_update[4]}')
else:
    print('No updates found')

print(f'\n=== VAL LOSSES ===')
for v in val_losses[-10:]:
    print(v)

print(f'\n=== BEST MODELS ===')
for b in best_models[-5:]:
    print(b)

print(f'\n=== ERRORS ({len(errors)} total) ===')
for e in errors[-10:]:
    print(e)

# R1 values
r1_vals = []
for line in lines:
    m = re.search(r'r1[=:]([0-9.e+]+)', line)
    if m:
        try:
            r1_vals.append(float(m.group(1)))
        except:
            pass

if r1_vals:
    print(f'\n=== R1 PENALTY STATS ===')
    print(f'Count: {len(r1_vals)}, Min: {min(r1_vals):.2f}, Max: {max(r1_vals):.2f}, Last: {r1_vals[-1]:.2f}')

print('\n=== TAIL OF LOG (last 500 chars) ===')
print(data[-500:])
