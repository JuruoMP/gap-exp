import json
import os

import _jsonnet

from seq2struct import datasets
from seq2struct.utils import registry


def post_process(infer_data, data_path):
    dialog_lens = []
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    for dialog in data:
        interactions = dialog['interaction']
        dialog_lens.append(len(interactions))

    with open('predicted_sql.txt', 'w', encoding='utf-8') as fw:
        for sql in infer_data:
            fw.write(sql + '\n')
        if dialog_lens[0] == 0:
            del dialog_lens[0]
            fw.write('\n')
        else:
            dialog_lens[0] -= 1

def compute_metrics(config_path, config_args, section, inferred_path,logdir=None):
    if config_args:
        config = json.loads(_jsonnet.evaluate_file(config_path, tla_codes={'args': config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(config_path))

    if 'model_name' in config and logdir:
        logdir = os.path.join(logdir, config['model_name'])
    if logdir:
        inferred_path = inferred_path.replace('__LOGDIR__', logdir)

    inferred = open(inferred_path)
    data = registry.construct('dataset', config['data'][section])
    metrics = data.Metrics(data)

    inferred_lines = list(inferred)
    if len(inferred_lines) < len(data):
        raise Exception('Not enough inferred: {} vs {}'.format(len(inferred_lines),
          len(data)))

    infer_data = []
    for line in inferred_lines:
        infer_results = json.loads(line)
        if infer_results['beams']:
            inferred_code = infer_results['beams'][0]['inferred_code']
            infer_data.append(inferred_code)
        else:
            inferred_code = None
        if 'index' in infer_results:
            metrics.add(data[infer_results['index']], inferred_code)
        else:
            metrics.add(None, inferred_code, obsolete_gold_code=infer_results['gold_code'])

    return logdir, metrics.finalize()
