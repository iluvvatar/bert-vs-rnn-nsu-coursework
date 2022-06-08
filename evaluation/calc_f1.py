from pathlib import Path
import os
import json
from pprint import pprint

from mlpipeline.datasets.nlp import RuREBus
from mlpipeline.datasets.nlp.units import Entity
from mlpipeline.evaluators.ner import EvaluatorNER



def main():
    home_dir = Path(os.getenv('HOME'))
    dataset_dir = home_dir / 'datasets' / 'RuREBus'
    rurebus_path = dataset_dir / 'HuggingFaceLocal'
    rurebus = RuREBus.load(rurebus_path)
    datasets_dir = dataset_dir / 'models_predictions'

    evaluator = EvaluatorNER(real_entities_column='entities',
                             pred_entities_column='predicted_entities',
                             entity_types=rurebus.entity_types,
                             entities_deserialize_fn=Entity.from_brat)

    results = []
    for ds_name in os.listdir(datasets_dir):
        ds_path = datasets_dir / ds_name
        ds = RuREBus.load(ds_path)
        evaluation_results = {'name': ds_name}
        evaluation_results.update(evaluator.evaluate(ds))
        print()
        print(ds_name)
        pprint(evaluation_results)
        results.append(evaluation_results)
    with open(datasets_dir / 'eval_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == '__main__':
    main()
