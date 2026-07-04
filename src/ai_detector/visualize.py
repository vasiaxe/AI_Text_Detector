import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ai_detector.utils import ensure_dir, load_config


def load_json(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def plot_confusion_matrix(results_dir: Path):
    path = results_dir / 'test_confusion_matrix.json'

    if not path.exists():
        return

    data = load_json(path)
    matrix = data['confusion_matrix']
    labels = data.get('labels', ['human', 'ai_generated'])

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(matrix)

    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                int(matrix[row][column]),
                ha='center',
                va='center'
            )

    fig.tight_layout()
    fig.savefig(results_dir / 'confusion_matrix.png', dpi=200)
    plt.close(fig)


def plot_training_history(results_dir: Path):
    path = results_dir / 'metrics.json'

    if not path.exists():
        return

    metrics = load_json(path)
    history = metrics.get('history', [])

    if not history:
        return

    epochs = [item['epoch'] for item in history]
    train_losses = [item['train_loss'] for item in history]
    validation_losses = [item['validation_loss'] for item in history]
    validation_f1 = [item['validation_f1'] for item in history]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(epochs, train_losses, label='train_loss')
    ax.plot(epochs, validation_losses, label='validation_loss')
    ax.plot(epochs, validation_f1, label='validation_f1')

    ax.set_title('Training History')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend()

    fig.tight_layout()
    fig.savefig(results_dir / 'training_history.png', dpi=200)
    plt.close(fig)


def plot_probability_distribution(predictions_df: pd.DataFrame, results_dir: Path):
    human_scores = predictions_df[
        predictions_df['true_label'] == 0
    ]['ai_probability']

    ai_scores = predictions_df[
        predictions_df['true_label'] == 1
    ]['ai_probability']

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        human_scores,
        bins=20,
        alpha=0.6,
        label='true_human'
    )

    ax.hist(
        ai_scores,
        bins=20,
        alpha=0.6,
        label='true_ai_generated'
    )

    ax.set_title('AI Probability Distribution')
    ax.set_xlabel('AI probability')
    ax.set_ylabel('Text count')
    ax.legend()

    fig.tight_layout()
    fig.savefig(results_dir / 'probability_distribution.png', dpi=200)
    plt.close(fig)


def plot_confidence_vs_error(predictions_df: pd.DataFrame, results_dir: Path):
    df = predictions_df.copy()

    df['confidence_bin'] = pd.cut(
        df['confidence'],
        bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        include_lowest=True
    )

    grouped = df.groupby('confidence_bin', observed=False)['is_correct']
    error_rate = 1 - grouped.mean()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        [str(index) for index in error_rate.index],
        error_rate.values
    )

    ax.set_title('Error Rate by Confidence')
    ax.set_xlabel('Confidence bin')
    ax.set_ylabel('Error rate')

    plt.xticks(rotation=35, ha='right')

    fig.tight_layout()
    fig.savefig(results_dir / 'confidence_vs_error.png', dpi=200)
    plt.close(fig)


def plot_error_by_text_length(predictions_df: pd.DataFrame, results_dir: Path):
    df = predictions_df.copy()

    df['length_bin'] = pd.cut(
        df['word_count'],
        bins=[0, 100, 250, 500, 1000, float('inf')],
        labels=['0-100', '101-250', '251-500', '501-1000', '1000+'],
        include_lowest=True
    )

    grouped = df.groupby('length_bin', observed=False)['is_correct']
    error_rate = 1 - grouped.mean()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        [str(index) for index in error_rate.index],
        error_rate.values
    )

    ax.set_title('Error Rate by Text Length')
    ax.set_xlabel('Word count')
    ax.set_ylabel('Error rate')

    fig.tight_layout()
    fig.savefig(results_dir / 'error_by_text_length.png', dpi=200)
    plt.close(fig)


def plot_error_by_column(
    predictions_df: pd.DataFrame,
    results_dir: Path,
    column: str,
    output_name: str,
    title: str
):
    if column not in predictions_df.columns:
        return

    df = predictions_df.copy()

    if df[column].nunique() < 2:
        return

    grouped = df.groupby(column)['is_correct']
    summary = pd.DataFrame({
        'error_rate': 1 - grouped.mean(),
        'count': grouped.count()
    })

    summary = summary[summary['count'] >= 5]
    summary = summary.sort_values('error_rate', ascending=False)

    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        summary.index.astype(str),
        summary['error_rate'].values
    )

    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel('Error rate')

    plt.xticks(rotation=35, ha='right')

    fig.tight_layout()
    fig.savefig(results_dir / output_name, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = ensure_dir(config['results_dir'])

    plot_training_history(results_dir)
    plot_confusion_matrix(results_dir)

    predictions_path = results_dir / 'predictions.csv'

    if not predictions_path.exists():
        return

    predictions_df = pd.read_csv(predictions_path)

    plot_probability_distribution(predictions_df, results_dir)
    plot_confidence_vs_error(predictions_df, results_dir)
    plot_error_by_text_length(predictions_df, results_dir)

    plot_error_by_column(
        predictions_df,
        results_dir,
        column='genre',
        output_name='error_by_genre.png',
        title='Error Rate by Genre'
    )

    plot_error_by_column(
        predictions_df,
        results_dir,
        column='model',
        output_name='error_by_model.png',
        title='Error Rate by Generator Model'
    )


if __name__ == '__main__':
    main()