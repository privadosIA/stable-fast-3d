## ADDED Requirements

### Requirement: The weights are read as the container has them

The library SHALL read the weights from the folder layout of the container's `/ckpt` (`sf3d/config.yaml, sf3d/model.safetensors, sf3d/dinov2-large-config.json, sf3d/160_tets.npz, rembg/u2net.onnx`) with no Python and no download, and SHALL refuse a file that is not the checkpoint it expects naming the key that is missing or left over. The C# modules SHALL carry the names of the checkpoint's keys so that loading needs no mapping. The only keys allowed to be left over are the ones the port names as unused: the CLIP text tower (the 150 tensors of `image_estimator.model` that are not `image_estimator.model.visual.*`: the checkpoint holds all of ViT-B/32 but the roughness and metallic heads read its vision tower only) and the global illumination estimator (the 12 tensors of `global_estimator.*`, not ported: `estimate_illumination` is off in the container).

#### Scenario: The strict load
- **WHEN** the checkpoint is loaded into the modules
- **THEN** every key is used once and none is left over except the named unused ones (the CLIP text tower and the global estimator); any other key that does not fit is named

#### Scenario: Nothing is downloaded
- **WHEN** the service starts with the weights folder and no network
- **THEN** it loads the model and makes a model

### Requirement: Checkpoints are never in the repository

The repository SHALL hold no weights; tests that need them SHALL be skipped, saying why, unless `SF3D_CKPT` points to them, and fixtures SHALL be checked against a sha256 table before use.

#### Scenario: A clone with no weights
- **WHEN** the tests run on a clone with no weights
- **THEN** they pass, and the ones that need the weights say they were skipped and why
