import os
from pathlib import Path
import json

from mlpipeline.datasets.nlp.units import Entity, Relation


def main():
    home_dir = Path(os.getenv('HOME'))
    dataset_dir = home_dir / 'datasets/RuREBus/HuggingFaceHub'
    data_dir = dataset_dir / 'data'
    out_ent_types_path = dataset_dir / 'entity_types.txt'
    out_rel_types_path = dataset_dir / 'relation_types.txt'
    ent_types = set()
    rel_types = set()
    with open(data_dir / 'train.jsonl', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            for e in line['entities']:
                e = Entity.from_brat(e)
                ent_types.add(e.type)
            for r in line['relations']:
                r = Relation.from_brat(r)
                rel_types.add(r.type)
    with open(data_dir / 'test.jsonl', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            for e in line['entities']:
                e = Entity.from_brat(e)
                ent_types.add(e.type)
            for r in line['relations']:
                r = Relation.from_brat(r)
                rel_types.add(r.type)
    ent_types = sorted(ent_types)
    rel_types = sorted(rel_types)
    print(ent_types)
    print(rel_types)
    with open(out_ent_types_path, 'w', encoding='utf-8') as f:
        for e in ent_types:
            print(e, file=f)
    with open(out_rel_types_path, 'w', encoding='utf-8') as f:
        for r in rel_types:
            print(r, file=f)


if __name__ == '__main__':
    main()
