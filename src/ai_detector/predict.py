import argparse
from pathlib import Path
import joblib
import numpy as np
import torch

from ai_detector.data import build_eval_style_features, tokenize_texts
from ai_detector.model import build_model, load_tokenizer
from ai_detector.utils import get_device, load_config


def predict_text(text: str, config: dict):
    device = get_device()
    output_dir = Path(config['output_dir'])

    tokenizer = load_tokenizer(output_dir / 'tokenizer')
    scaler = joblib.load(output_dir / 'scaler.joblib')

    inputs = tokenize_texts(
        [text],
        tokenizer,
        config['max_length']
    )

    style_features = build_eval_style_features(
        [text],
        scaler,
        config.get('style_clip_value', 5)
    )

    model = build_model(config).to(device)
    model.load_state_dict(
        torch.load(
            output_dir / 'best_model_state.pt',
            map_location=device
        )
    )

    model.eval()

    with torch.no_grad():
        logits = model(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            style_features=style_features.to(device)
        )

        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_label = int(np.argmax(probabilities))

    return {
        'prediction': predicted_label,
        'human_probability': float(probabilities[0]),
        'ai_probability': float(probabilities[1])
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    result = predict_text(args.text, config)

    label = 'ai_generated' if result['prediction'] == 1 else 'human'

    print(f'Prediction: {label}')
    print(f'Human probability: {result["human_probability"]:.4f}')
    print(f'AI probability: {result["ai_probability"]:.4f}')


if __name__ == '__main__':
    main()