import os
from pathlib import Path
from pprint import pprint
import json

from evaluation.original_scripts.evaluate_ners import calc_ner_f1

home_dir = Path(os.getenv('HOME'))
dataset_dir = home_dir / 'datasets' / 'RuREBus'
real_dir = dataset_dir / 'real_test_docs_brat'
pred_root = dataset_dir / 'models_predictions_united_docs_brat'
out_file = dataset_dir / 'f1_micro_test (original eval scripts).json'

results = {}
for pred_dir in pred_root.iterdir():
    results[pred_dir.name] = calc_ner_f1(real_dir, pred_root / pred_dir)
pprint(results)
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False)
