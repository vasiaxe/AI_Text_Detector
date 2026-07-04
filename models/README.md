# Model artifacts

Trained model artifacts are not committed to this repository.

The final trained model directory contains:

```text
models/ai_detector/best_model_state.pt
models/ai_detector/scaler.joblib
models/ai_detector/tokenizer/
models/ai_detector/train_config.yaml
```

These files are excluded because the checkpoint is large and not suitable for normal GitHub storage.

The public repository contains the training code, evaluation code, configuration, metrics, and plots instead.