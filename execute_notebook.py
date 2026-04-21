#!/usr/bin/env python3.14
"""Execute notebook cells, skipping the training loop cell."""

import json
import copy
import os

import nbformat
from nbclient import NotebookClient

NB_PATH = "HAND_JOB/gesture_classifier.ipynb"
TRAINING_LOOP_MARKER = "for epoch in range(EPOCHS):"

with open(NB_PATH) as f:
    nb = nbformat.read(f, as_version=4)

# Find training loop cell and temporarily replace its source with 'pass'
training_loop_idx = None
training_loop_source = None

for i, cell in enumerate(nb.cells):
    src = "".join(cell["source"])
    if TRAINING_LOOP_MARKER in src:
        training_loop_idx = i
        training_loop_source = cell["source"]
        print(f"Found training loop cell at index {i}, temporarily replacing with 'pass'")
        cell["source"] = "pass  # training loop skipped — using pre-trained checkpoint"
        break

assert training_loop_idx is not None, "Training loop cell not found!"

# Execute the notebook with nbclient
print(f"Executing notebook (working dir: HAND_JOB/)...")
client = NotebookClient(
    nb,
    timeout=600,
    kernel_name="python3",
    allow_errors=False,
    resources={"metadata": {"path": "HAND_JOB/"}},
)

try:
    client.execute()
    print("Notebook execution complete!")
except Exception as e:
    print(f"ERROR during execution: {e}")
    raise

# Restore training loop cell source, clear outputs
nb.cells[training_loop_idx]["source"] = training_loop_source
nb.cells[training_loop_idx]["outputs"] = []
nb.cells[training_loop_idx]["execution_count"] = None
print(f"Restored training loop cell source and cleared outputs")

# Save executed notebook
with open(NB_PATH, "w") as f:
    nbformat.write(nb, f)
print(f"Saved executed notebook to {NB_PATH}")

# Print test macro-F1 from Task 14 cell output
task14_idx = None
for i, cell in enumerate(nb.cells):
    src = "".join(cell["source"])
    if "Task 14" in src and "test_loader" in src:
        task14_idx = i
        break

if task14_idx is not None:
    cell = nb.cells[task14_idx]
    for output in cell.get("outputs", []):
        text = output.get("text", "")
        if isinstance(text, list):
            text = "".join(text)
        if "TEST" in text or "macro" in text.lower():
            print(f"\n=== Task 14 output (first 2000 chars) ===")
            print(text[:2000])
            break
