import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(
        path,
        orient='records',
        lines=True,
        force_ascii=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--output-dir', type=str, default='data/raw')
    parser.add_argument('--text-column', type=str, default='text')
    parser.add_argument('--label-column', type=str, default='label')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    dataframes = [
        load_dataframe(Path(path))
        for path in args.input
    ]

    df = pd.concat(dataframes, ignore_index=True)

    df = df.dropna(subset=[args.text_column, args.label_column])
    df = df.drop_duplicates(subset=[args.text_column])
    df = df.reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=args.seed,
        stratify=df[args.label_column]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=args.seed,
        stratify=temp_df[args.label_column]
    )

    save_jsonl(train_df, output_dir / 'train.jsonl')
    save_jsonl(val_df, output_dir / 'val.jsonl')
    save_jsonl(test_df, output_dir / 'test.jsonl')

    print(f'Train: {len(train_df)}')
    print(f'Validation: {len(val_df)}')
    print(f'Test: {len(test_df)}')


if __name__ == '__main__':
    main()