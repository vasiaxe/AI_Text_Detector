#!/bin/bash

python -m ai_detector.predict \
  --config configs/train_config.yaml \
  --text 'This is a short example text to classify'