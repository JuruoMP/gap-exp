import json
import attr
import copy

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import networkx as nx
from seq2seq.lightning import evaluation
from seq2seq.utils.dataset import normalize, serialize_schema
from seq2seq.utils.cosql import cosql_get_input


@attr.s
class SparcItem:
    id = attr.ib()
    text = attr.ib()
    code = attr.ib()
    schema = attr.ib()
    orig = attr.ib()
    orig_schema = attr.ib()


@attr.s
class Column:
    id = attr.ib()
    table = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    type = attr.ib()
    foreign_key_for = attr.ib(default=None)


@attr.s
class Table:
    id = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    columns = attr.ib(factory=list)
    primary_keys = attr.ib(factory=list)


@attr.s
class Schema:
    db_id = attr.ib()
    tables = attr.ib()
    columns = attr.ib()
    foreign_key_graph = attr.ib()
    orig = attr.ib()


def load_tables(path):
    schemas = {}
    eval_foreign_key_maps = {}

    schema_dicts = json.load(open(path))
    for schema_dict in schema_dicts:
        tables = tuple(
            Table(
                id=i,
                name=name.split(),
                unsplit_name=name,
                orig_name=orig_name,
            )
            for i, (name, orig_name) in enumerate(zip(
                schema_dict['table_names'], schema_dict['table_names_original']))
        )
        columns = tuple(
            Column(
                id=i,
                table=tables[table_id] if table_id >= 0 else None,
                name=col_name.split(),
                unsplit_name=col_name,
                orig_name=orig_col_name,
                type=col_type,
            )
            for i, ((table_id, col_name), (_, orig_col_name), col_type) in enumerate(zip(
                schema_dict['column_names'],
                schema_dict['column_names_original'],
                schema_dict['column_types']))
        )

        # Link columns to tables
        for column in columns:
            if column.table:
                column.table.columns.append(column)

        for column_id in schema_dict['primary_keys']:
            # Register primary keys
            column = columns[column_id]
            column.table.primary_keys.append(column)

        foreign_key_graph = nx.DiGraph()
        for source_column_id, dest_column_id in schema_dict['foreign_keys']:
            # Register foreign keys
            source_column = columns[source_column_id]
            dest_column = columns[dest_column_id]
            source_column.foreign_key_for = dest_column
            foreign_key_graph.add_edge(
                source_column.table.id,
                dest_column.table.id,
                columns=(source_column_id, dest_column_id))
            foreign_key_graph.add_edge(
                dest_column.table.id,
                source_column.table.id,
                columns=(dest_column_id, source_column_id))

        db_id = schema_dict['db_id']
        assert db_id not in schemas
        schemas[db_id] = Schema(db_id, tables, columns, foreign_key_graph, schema_dict)
        eval_foreign_key_maps[db_id] = evaluation.build_foreign_key_map(schema_dict)

    return schemas, eval_foreign_key_maps


class SparcDataset(torch.utils.data.Dataset):
    def __init__(self, path, tables_paths, db_path, tokenizer):
        self.path = path
        self.db_path = db_path
        self.examples = []
        self.use_column_type = False
        self.tokenizer = tokenizer
        self.max_seq_len = self.tokenizer.model_max_length

        self.schemas, self.eval_foreign_key_maps = load_tables(tables_paths)

        self.normalize_query = True
        self.use_serialized_schema = True

        raw_data = json.load(open(path))
        for entry in raw_data:
            accumulated_toks = []
            for i, interaction in enumerate(entry['interaction']):
                new_toks = interaction['utterance_toks']
                accumulated_toks.append(new_toks)
                item = SparcItem(
                    id=len(self.examples),
                    text=copy.deepcopy(accumulated_toks),
                    code=interaction['query'],
                    schema=self.schemas[entry['database_id']],
                    orig=(entry, i),
                    orig_schema=self.schemas[entry['database_id']].orig)
                if self.validate_item(item):
                    self.examples.append(item)

        print('Sparc dataset built.')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = copy.deepcopy(self.examples[idx])
        item.text = item.text[1:] + item.text[:1]  # strategy1: bring current utterance to the front
        # item.text = item.text[::-1]  # strategy2: reverse text
        encoder_dict, decoder_dict = self.tokenize_item(item)
        labels = [x if x != self.tokenizer.pad_token_id else -100 for x in decoder_dict['input_ids']]
        return {
            'id': item.id,
            'input_ids': encoder_dict['input_ids'],
            'attention_mask': encoder_dict['attention_mask'],
            # 'decoder_input_ids': decoder_dict['input_ids'],
            'decoder_attention_mask': decoder_dict['attention_mask'],
            'labels': labels,
        }

    def tokenize_item(self, item):
        nl = ' '.join([t for s in item.text for t in s])
        sql = normalize(item.code) if self.normalize_query else item.code
        if self.use_serialized_schema:
            serialized_schema = serialize_schema(
                question=nl,
                db_path=self.db_path,
                db_id=item.orig[0]['database_id'],
                db_table_names=item.schema.orig['table_names'],
                db_column_names={
                    'table_id': [x[0] for x in item.schema.orig['column_names_original']],
                    'column_name': [x[1] for x in item.schema.orig['column_names_original']]
                },
                schema_serialization_type="peteshaw",
                schema_serialization_randomized=False,
                schema_serialization_with_db_id=True,
                schema_serialization_with_db_content=False
            )
        else:
            serialized_schema = ''
            columns = []
            for c in item.schema.columns:
                if c and c.table:
                    tn, cn = c.table.orig_name, c.orig_name
                    columns.append((tn, cn))
            for c in columns:
                serialized_schema += c[0] + ' ' + c[1] + ' # '
            serialized_schema = serialized_schema.rstrip(' # ')
        concat_input = cosql_get_input(
            utterances=[' '.join(x) for x in item.text],
            serialized_schema=serialized_schema,
            prefix=''
        )
        encoder_dict = self.tokenizer(concat_input + ' </s>')
        decoder_dict = self.tokenizer(sql + ' </s>')
        return encoder_dict, decoder_dict

    def validate_item(self, item):
        encoder_dict, decoder_dict = self.tokenize_item(item)
        return len(encoder_dict['input_ids']) < self.max_seq_len and len(decoder_dict['input_ids']) < self.max_seq_len

    def collate_fn(self, x_list):
        max_input_len = max(len(x['input_ids']) for x in x_list)
        max_output_len = max(len(x['labels']) for x in x_list)
        for x in x_list:
            x['input_ids'] += [0 for _ in range(max_input_len - len(x['input_ids']))]
            x['attention_mask'] += [0 for _ in range(max_input_len - len(x['attention_mask']))]
            # x['decoder_input_ids'] += [0 for _ in range(max_output_len - len(x['decoder_input_ids']))]
            x['decoder_attention_mask'] += [0 for _ in range(max_output_len - len(x['decoder_attention_mask']))]
            x['labels'] += [-100 for _ in range(max_output_len - len(x['labels']))]
        return default_collate([{k: torch.tensor(v).long() for k, v in x.items()} for x in x_list])

    class Metrics:
        def __init__(self, dataset):
            self.dataset = dataset
            self.foreign_key_maps = {
                db_id: evaluation.build_foreign_key_map(schema.orig)
                for db_id, schema in self.dataset.schemas.items()
            }
            self.evaluator = evaluation.Evaluator(
                self.dataset.db_path,
                self.foreign_key_maps,
                'match')
            self.results = []

        def add(self, item, inferred_code, orig_question=None):
            ret_dict = self.evaluator.evaluate_one(
                item.schema.db_id, item.orig['query'], inferred_code)
            if orig_question:
                ret_dict["orig_question"] = orig_question
            self.results.append(ret_dict)

        def add_beams(self, item, inferred_codes, orig_question=None):
            beam_dict = {}
            if orig_question:
                beam_dict["orig_question"] = orig_question
            for i, code in enumerate(inferred_codes):
                ret_dict = self.evaluator.evaluate_one(
                    item.schema.db_id, item.orig['query'], code)
                beam_dict[i] = ret_dict
                if ret_dict["exact"] is True:
                    break
            self.results.append(beam_dict)

        def finalize(self):
            self.evaluator.finalize()
            return {
                'per_item': self.results,
                'total_scores': self.evaluator.scores
            }


if __name__ == '__main__':
    train_data = SparcDataset('sparc/train.json', 'sparc/tables.json', 'sparc/database')
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=7, collate_fn=train_data.collate_fn)
    for batch in dataloader:
        a = 1