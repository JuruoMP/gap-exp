import sys
sys.path.append('./')

import json

from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.hf_argparser import HfArgumentParser
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments

from seq2seq.utils.args import ModelArguments
from seq2seq.utils.dataset import DataTrainingArguments, DataArguments
from seq2seq.utils.dataset_loader import load_dataset
from seq2seq.trainer import SpiderTrainer, CoSQLTrainer

def main():
    config_file = 'seq2seq/configs/train_sparc.json'

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    model_args: ModelArguments
    data_args: DataArguments
    data_training_args: DataTrainingArguments
    training_args: Seq2SeqTrainingArguments

    model_args, data_args, data_training_args, training_args = parser.parse_json_file(
        json_file=config_file
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        max_length=data_training_args.max_target_length,
        num_beams=data_training_args.num_beams,
        num_beam_groups=data_training_args.num_beam_groups,
        diversity_penalty=data_training_args.diversity_penalty,
        gradient_checkpointing=model_args.gradient_checkpointing,
        use_cache=not model_args.gradient_checkpointing,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    metric, dataset_splits = load_dataset(
        data_args=data_args,
        model_args=model_args,
        data_training_args=data_training_args,
        training_args=training_args,
        tokenizer=tokenizer,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "metric": metric,
        "train_dataset": dataset_splits.train_split.dataset if training_args.do_train else None,
        "eval_dataset": dataset_splits.eval_split.dataset if training_args.do_eval else None,
        "eval_examples": dataset_splits.eval_split.examples if training_args.do_eval else None,
        "tokenizer": tokenizer,
        "data_collator": DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=(-100 if data_training_args.ignore_pad_token_for_loss else tokenizer.pad_token_id),
            pad_to_multiple_of=8 if training_args.fp16 else None,
        ),
        "ignore_pad_token_for_loss": data_training_args.ignore_pad_token_for_loss,
        "target_with_db_id": data_training_args.target_with_db_id,
    }
    trainer = CoSQLTrainer(**trainer_kwargs)
    train_result = trainer.train()
    trainer.save_model()
    metrics = train_result.metrics
    print(metrics)


if __name__ == '__main__':
    main()
