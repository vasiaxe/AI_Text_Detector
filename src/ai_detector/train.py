import argparse
import copy
import shutil
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from tqdm import tqdm

from ai_detector.data import create_train_val_dataloaders
from ai_detector.model import build_model, build_optimizer, load_tokenizer
from ai_detector.utils import ensure_dir, get_device, load_config, save_json, set_seed


def train_one_epoch(model, train_loader, optimizer, criterion, device, max_grad_norm):
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc='Training', leave=False)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        style_features = batch['style_features'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            style_features=style_features
        )

        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm
        )

        if not torch.isfinite(grad_norm):
            raise RuntimeError('Non-finite gradient norm')

        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating', leave=False):
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

            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite validation loss')

            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average='binary',
        zero_division=0
    )

    matrix = confusion_matrix(all_labels, all_predictions)

    return {
        'loss': float(avg_loss),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': matrix.tolist()
    }


def save_training_artifacts(
    model,
    tokenizer,
    scaler,
    config,
    config_path,
    output_dir,
    results_dir,
    best_metrics,
    history
):
    ensure_dir(output_dir)
    ensure_dir(results_dir)

    torch.save(
        model.state_dict(),
        output_dir / 'best_model_state.pt'
    )

    joblib.dump(
        scaler,
        output_dir / 'scaler.joblib'
    )

    tokenizer.save_pretrained(
        output_dir / 'tokenizer'
    )

    shutil.copy2(
        config_path,
        output_dir / 'train_config.yaml'
    )

    metrics = {
        'project_name': config['project_name'],
        'model_name': config['model_name'],
        'best_epoch': best_metrics['epoch'],
        'best_validation_loss': best_metrics['loss'],
        'best_validation_accuracy': best_metrics['accuracy'],
        'best_validation_precision': best_metrics['precision'],
        'best_validation_recall': best_metrics['recall'],
        'best_validation_f1': best_metrics['f1'],
        'history': history
    }

    save_json(metrics, results_dir / 'metrics.json')

    save_json(
        {
            'confusion_matrix': best_metrics['confusion_matrix'],
            'labels': ['human', 'ai_generated']
        },
        results_dir / 'validation_confusion_matrix.json'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    set_seed(config['random_seed'])

    device = get_device()
    output_dir = Path(config['output_dir'])
    results_dir = Path(config['results_dir'])

    tokenizer = load_tokenizer(config['model_name'])

    train_loader, val_loader, scaler = create_train_val_dataloaders(
        config,
        tokenizer
    )

    model = build_model(config).to(device)
    model = model.float()

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)

    best_model_state = None
    best_metrics = None
    best_f1 = -1
    history = []

    for epoch in range(1, config['epochs'] + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            config.get('max_grad_norm', 1.0)
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        epoch_metrics = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'validation_loss': val_metrics['loss'],
            'validation_accuracy': val_metrics['accuracy'],
            'validation_precision': val_metrics['precision'],
            'validation_recall': val_metrics['recall'],
            'validation_f1': val_metrics['f1']
        }

        history.append(epoch_metrics)

        print(
            f'Epoch {epoch}/{config["epochs"]} | '
            f'train_loss={train_loss:.4f} | '
            f'val_loss={val_metrics["loss"]:.4f} | '
            f'val_f1={val_metrics["f1"]:.4f}'
        )

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_model_state = copy.deepcopy(model.state_dict())
            best_metrics = {
                'epoch': epoch,
                **val_metrics
            }

    if best_model_state is None:
        raise RuntimeError('No valid model checkpoint was produced')

    model.load_state_dict(best_model_state)

    save_training_artifacts(
        model=model,
        tokenizer=tokenizer,
        scaler=scaler,
        config=config,
        config_path=config_path,
        output_dir=output_dir,
        results_dir=results_dir,
        best_metrics=best_metrics,
        history=history
    )


if __name__ == '__main__':
    main()