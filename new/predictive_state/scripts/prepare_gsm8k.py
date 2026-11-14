"""Export ONLY pinned GSM8K training questions from AKR's downloaded parquet.
No answers are read from the parquet table or written to the output JSONL.
No silent download, train/test mixing, or truncation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-parquet', required=True,
                   help='AKR pinned local data/gsm8k/main/train-00000-of-00001.parquet')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    import pyarrow.parquet as pq
    src = Path(args.train_parquet)
    dest = Path(args.out)
    if not src.is_file() or dest.exists():
        raise FileExistsError('source must exist and output must not preexist')
    table = pq.read_table(src, columns=['question'])
    if table.num_rows != 7473:
        raise ValueError('expected complete pinned GSM8K training split (7473)')
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x', encoding='utf-8') as stream:
        for index, row in enumerate(table.to_pylist()):
            question = row['question']
            if not isinstance(question, str) or not question.strip():
                raise ValueError('empty question')
            stream.write(json.dumps({'id': f'gsm8k_train_{index:05d}',
                                     'prompt': question}, ensure_ascii=False) + '\n')
    print(f'Exported {table.num_rows} question-only records: {dest}')


if __name__ == '__main__':
    main()
