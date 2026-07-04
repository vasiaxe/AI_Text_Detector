import argparse
from pathlib import Path
import joblib
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from torch import nn
from tqdm import tqdm

from ai_detector.data import create_eval_dataloader, load_dataframe
from ai_detector.model import build_model, load_tokenizer
from ai_detector.utils import ensure_dir, get_device, load_config, save_json


def run_evaluation(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0
    all_predictions = []
    all_labels = []
    all_human_probabilities = []
    all_ai_probabilities = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            style_features = batch['style_features'].to(device)
            labels = batch['labels'].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                style_features=style_features
            )

            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            total_loss += loss.item()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_human_probabilities.extend(probabilities[:, 0].cpu().numpy())
            all_ai_probabilities.extend(probabilities[:, 1].cpu().numpy())

    avg_loss = total_loss / len(data_loader)

    accuracy = accuracy_score(all_labels, all_predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average='binary',
        zero_division=0
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1]
    )

    metrics = {
        'test_loss': float(avg_loss),
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'confusion_matrix': matrix.tolist()
    }

    return {
        'metrics': metrics,
        'labels': all_labels,
        'predictions': all_predictions,
        'human_probabilities': all_human_probabilities,
        'ai_probabilities': all_ai_probabilities,
        'confusion_matrix': matrix
    }


def save_classification_report(labels, predictions, results_dir: Path):
    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=['human', 'ai_generated'],
        zero_division=0
    )

    with open(results_dir / 'classification_report.txt', 'w', encoding='utf-8') as file:
        file.write(report)


def load_prediction_metadata(config: dict) -> pd.DataFrame:
    text_column = config['text_column']
    label_column = config['label_column']

    df = load_dataframe(config['test_path'])
    df = df.dropna(subset=[text_column, label_column]).reset_index(drop=True)

    metadata = pd.DataFrame()
    metadata['text'] = df[text_column].astype(str)
    metadata['word_count'] = metadata['text'].str.split().str.len()

    for column in ['genre', 'model', 'source', 'dataset']:
        if column in df.columns:
            metadata[column] = df[column].astype(str)

    return metadata


def save_predictions_csv(evaluation_output: dict, config: dict, results_dir: Path):
    metadata = load_prediction_metadata(config)

    predictions_df = metadata.copy()
    predictions_df['true_label'] = evaluation_output['labels']
    predictions_df['predicted_label'] = evaluation_output['predictions']
    predictions_df['human_probability'] = evaluation_output['human_probabilities']
    predictions_df['ai_probability'] = evaluation_output['ai_probabilities']

    predictions_df['confidence'] = predictions_df[
        ['human_probability', 'ai_probability']
    ].max(axis=1)

    predictions_df['is_correct'] = (
        predictions_df['true_label'] == predictions_df['predicted_label']
    )

    predictions_df['true_label_name'] = predictions_df['true_label'].map({
        0: 'human',
        1: 'ai_generated'
    })

    predictions_df['predicted_label_name'] = predictions_df['predicted_label'].map({
        0: 'human',
        1: 'ai_generated'
    })

    predictions_df.to_csv(
        results_dir / 'predictions.csv',
        index=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    device = get_device()
    output_dir = Path(config['output_dir'])
    results_dir = ensure_dir(config['results_dir'])

    tokenizer_path = output_dir / 'tokenizer'
    model_path = output_dir / 'best_model_state.pt'
    scaler_path = output_dir / 'scaler.joblib'

    tokenizer = load_tokenizer(str(tokenizer_path))
    scaler = joblib.load(scaler_path)

    data_loader = create_eval_dataloader(
        config['test_path'],
        config,
        tokenizer,
        scaler
    )

    model = build_model(config).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    criterion = nn.CrossEntropyLoss()

    evaluation_output = run_evaluation(
        model,
        data_loader,
        criterion,
        device
    )

    save_json(evaluation_output['metrics'], results_dir / 'test_metrics.json')

    save_json(
        {
            'confusion_matrix': evaluation_output['confusion_matrix'].tolist(),
            'labels': ['human', 'ai_generated']
        },
        results_dir / 'test_confusion_matrix.json'
    )

    save_classification_report(
        evaluation_output['labels'],
        evaluation_output['predictions'],
        results_dir
    )

    save_predictions_csv(
        evaluation_output,
        config,
        results_dir
    )


if __name__ == '__main__':
    main()