from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from ai_detector.features import extract_stylometry_batch


class FusionDataset(Dataset):
    def __init__(self, inputs, style_features, labels):
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']
        self.style_features = style_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_mask[index],
            'style_features': self.style_features[index],
            'labels': self.labels[index]
        }


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f'Cannot find data file: {path}')

    if path.suffix == '.jsonl':
        return pd.read_json(path, lines=True)

    if path.suffix == '.csv':
        return pd.read_csv(path)

    raise ValueError(f'Unsupported data format: {path.suffix}')


def load_texts_and_labels(
    path: str | Path,
    text_column: str,
    label_column: str
) -> tuple[list[str], torch.Tensor]:
    df = load_dataframe(path)

    missing_columns = {text_column, label_column} - set(df.columns)

    if missing_columns:
        raise ValueError(f'Missing columns: {missing_columns}')

    df = df[[text_column, label_column]].dropna()
    texts = df[text_column].astype(str).tolist()
    labels = torch.tensor(df[label_column].astype(int).tolist(), dtype=torch.long)

    return texts, labels


def tokenize_texts(texts: list[str], tokenizer, max_length: int):
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )


def build_train_val_style_features(
    train_texts: list[str],
    val_texts: list[str],
    clip_value: float
):
    train_features = extract_stylometry_batch(train_texts)
    val_features = extract_stylometry_batch(val_texts)

    scaler = StandardScaler()

    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)

    train_features = np.clip(train_features, -clip_value, clip_value)
    val_features = np.clip(val_features, -clip_value, clip_value)

    train_features = torch.tensor(train_features, dtype=torch.float32)
    val_features = torch.tensor(val_features, dtype=torch.float32)

    return train_features, val_features, scaler


def build_eval_style_features(
    texts: list[str],
    scaler,
    clip_value: float
) -> torch.Tensor:
    features = extract_stylometry_batch(texts)
    features = scaler.transform(features)
    features = np.clip(features, -clip_value, clip_value)

    return torch.tensor(features, dtype=torch.float32)


def create_train_val_dataloaders(config: dict, tokenizer):
    text_column = config['text_column']
    label_column = config['label_column']
    max_length = config['max_length']
    batch_size = config['batch_size']
    clip_value = config.get('style_clip_value', 5)

    train_texts, train_labels = load_texts_and_labels(
        config['train_path'],
        text_column,
        label_column
    )

    val_texts, val_labels = load_texts_and_labels(
        config['val_path'],
        text_column,
        label_column
    )

    train_inputs = tokenize_texts(train_texts, tokenizer, max_length)
    val_inputs = tokenize_texts(val_texts, tokenizer, max_length)

    train_style_features, val_style_features, scaler = build_train_val_style_features(
        train_texts,
        val_texts,
        clip_value
    )

    train_dataset = FusionDataset(
        train_inputs,
        train_style_features,
        train_labels
    )

    val_dataset = FusionDataset(
        val_inputs,
        val_style_features,
        val_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, scaler


def create_eval_dataloader(
    data_path: str | Path,
    config: dict,
    tokenizer,
    scaler
):
    text_column = config['text_column']
    label_column = config['label_column']
    max_length = config['max_length']
    batch_size = config['eval_batch_size']
    clip_value = config.get('style_clip_value', 5)

    texts, labels = load_texts_and_labels(
        data_path,
        text_column,
        label_column
    )

    inputs = tokenize_texts(texts, tokenizer, max_length)
    style_features = build_eval_style_features(texts, scaler, clip_value)

    dataset = FusionDataset(
        inputs,
        style_features,
        labels
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )