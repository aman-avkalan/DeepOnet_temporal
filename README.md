# DeepONet Training for Lid-Driven Cavity Flow using Temporal

## Overview

This project implements a **Deep Operator Network (DeepONet)** to learn the solution operator of the **2D Lid-Driven Cavity (LDC)** problem governed by the incompressible Navier–Stokes equations.  
The training and evaluation pipeline is orchestrated using **Temporal**, enabling fault-tolerant, restartable, and scalable execution of long-running GPU-based training jobs.

The goal is to map **geometry and flow descriptors** to **full-field velocity and pressure solutions** on a fixed grid.

---

## Problem Definition: Lid-Driven Cavity (LDC)

The lid-driven cavity problem consists of a square cavity where:
- The top lid moves with constant horizontal velocity
- Other walls are stationary
- The flow develops vortices and pressure gradients inside the cavity

This problem is widely used as a benchmark for:
- Computational fluid dynamics
- Operator learning
- Physics-informed and data-driven surrogate models

---

## Dataset Description

### Input Dataset (X)

Each sample in the input dataset has shape:

- (N, 3, H, W)

Channels:
1. **Reynolds number (Re)**  
   Encodes the flow regime.
2. **Signed Distance Function (SDF)**  
   Represents geometry information.
3. **Mask**  
   Distinguishes fluid and boundary regions.

### Output Dataset (Y)

Each output sample has shape:
- (N, 4, H, W)


Channels:
1. **u** – horizontal velocity
2. **v** – vertical velocity
3. **p** – pressure
4. **auxiliary / unused channel**

---

## DeepONet Architecture

The model follows the classical DeepONet formulation:

### Branch Network
- Input: full spatial fields (Re, SDF, Mask)
- Implemented as a CNN
- Produces a latent representation encoding global flow information

### Trunk Network
- Input: spatial coordinates `(x, y)`
- Implemented as an MLP
- Encodes location-dependent basis functions

### Operator Fusion
- Branch and trunk outputs are combined using an outer-product formulation
- A linear projection maps the fused representation to physical outputs
- Final output is reshaped into full spatial fields `(u, v, p)`

This formulation allows the network to learn a **mapping between function spaces**, not just pointwise regression.

---

## Coordinate-Based Learning

Instead of predicting outputs directly per grid cell, the model:
- Builds a normalized `(x, y)` coordinate grid
- Feeds coordinates into the trunk network
- Evaluates the operator at all spatial locations simultaneously

This enables:
- Resolution-aware learning
- Operator generalization across spatial domains

---

## Post-Training Visualization

After training, the activity performs inference on a single sample and generates a comprehensive diagnostic plot containing:

- Ground-truth velocity streamlines
- Predicted velocity streamlines
- Pressure fields
- Velocity error magnitude
- Absolute pressure error

All plots are generated using **NumPy-only arrays** to ensure compatibility with Matplotlib and Temporal execution.

The final visualization is saved as a PNG file inside the `outputs/` directory.

---

## Outputs

The project produces:
- A trained DeepONet model (in-memory)
- Epoch-wise training logs
- A saved PNG image comparing:
  - Ground truth vs predicted flow fields
  - Velocity and pressure errors

These outputs allow both **quantitative** and **qualitative** assessment of operator learning performance.

---

## How to Run

### 1. Start Temporal Server

`temporal server start-dev`

---

### 2. Start Worker (GPU node)

`python worker.py`

---

### 3. Run Workflow

`python run_workflow.py`

