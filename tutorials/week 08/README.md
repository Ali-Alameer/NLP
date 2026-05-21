# Secure and Trust-Aware Active Learning in Machine Vision

This repository contains the code, saved model artifacts, result logs, and figures for a study on adversarial labelling attacks in active learning for computer vision.

The full paper-style write-up is here:

- Project paper: [`readme_paper.md`](./paper/readme_paper.md)

## Repository Structure

- `readme_paper.md`
  Full dissertation-style project write-up, including methodology, results, and discussion.

- `paper/diagram/`
  Figures used by the paper, including workflow diagrams, model comparison figures, accuracy plots, and confusion matrices.

- `EXP/`
  Main experiment outputs, grouped by model track.

## Experiment Folders

The `EXP/` directory contains three experiment tracks:

- `EXP/EXP 1/`
  CoATNet-based experiment track.

- `EXP/EXP 2/`
  CvT-13-based experiment track.

- `EXP/EXP 3/`
  DeiT-Tiny-based experiment track.

Each experiment folder follows the same general structure:

- `code/`
  Jupyter notebooks for the staged pipeline:
  `Experiment Part A1`, `A2`, `A3`, and `B1`, plus a model-specific modified `B1` notebook for that track.

- `E/models/baselines/`
  Saved clean baseline model checkpoints and supporting baseline artifacts such as:
  classification reports, confusion analysis, architecture text summaries, and architecture diagrams.

- `E/results/`
  Final experiment outputs for the clean baseline and attack scenarios, including:
  accuracy logs, infection logs, labeled-size logs, classification reports, confusion matrices, and comparison plots.

## Experiment Pipeline

Across the three tracks, the notebooks and saved outputs follow this staged workflow:

- `A1`
  Clean baseline training and confusion mining.

- `A2`
  Confusion-focused subset creation and poisoned scenario construction.

- `A3`
  Scenario validation, inspection, and preparation.

- `B1`
  Active-learning execution and round-wise trust and utility logging.

## Figures

If you want the supporting images used in that write-up, see:

- [`diagram/`](./diagram)
