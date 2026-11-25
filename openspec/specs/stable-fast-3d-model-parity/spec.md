# stable-fast-3d-model-parity Specification

## Purpose

The model the port makes is the one the Python reference makes (Stable Fast 3D at commit `ff21fc4`, inside `veritmo/model3d:1`): every stage is checked against fixtures of a golden run with fixed thresholds, and the finished model is checked against the container's own.

## Requirements

### Requirement: Every stage matches the reference

Each stage of the model (see `docs/porting-plan.md`) SHALL be tested against fixtures of the golden run, in fp32, with `max_abs <= 1e-4` and `cos_sim >= 0.9999`, so that a difference is found at the stage where it starts; a change that touches the model SHALL NOT be archived without its parity test.

#### Scenario: A stage
- **WHEN** a stage runs on the fixtures' input
- **THEN** its output is within the thresholds of the fixture of the golden run

### Requirement: The model is the one the container makes

Over the five reference pictures, the model SHALL match the container's: the triangles count within 1% at each detail, the same bounding box within 1% of its size and a Chamfer distance below 1e-3 of it, upright (Y up) with its origin at the middle of its base, its colours kept, no more than 500,000 triangles (a model over them is a failed job with a sentence naming the count, as in the container: the mesh does not depend on the detail, which only sets the size of the baked textures, so making it again at a lower detail would not reduce it), and a `.glb` that the orchestrator reads (`blender.ParseModel`).

#### Scenario: The five pictures
- **WHEN** the five pictures go through the service at `normal` detail
- **THEN** each model matches the container's within those limits and stands on the floor of a scene

#### Scenario: Too many triangles
- **WHEN** a model passes 500,000 triangles
- **THEN** the job fails with a sentence naming the count; there is no retry, because the mesh is the same at every detail
