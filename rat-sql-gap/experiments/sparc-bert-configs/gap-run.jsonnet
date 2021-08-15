{
    local exp_id = 11,
    logdir: "logdir/sparc_bert_run_%d" %exp_id,
    model_config: "configs/sparc_bert_gap/gap-bert.jsonnet",
    model_config_args: {
        bs: 12,
        num_batch_accumulated: 2,
        bert_version: "bert-large-uncased",
        summarize_header: "avg",
        use_column_type: false,
        max_steps: 101000,
        num_layers: 8,
        lr: 1e-4,
        bert_lr: 1e-5,
        att: 1,
        end_lr: 0,
        sc_link: true,
        cv_link: true,
        use_align_mat: true,
        use_align_loss: true,
        bart_token_type: true,
        decoder_hidden_size: 512,
        end_with_from: true, # equivalent to "SWGOIF" if true
        clause_order: null, # strings like "SWGOIF", it will be prioriotized over end_with_from
    },

    eval_name: "sparc_bert_run_%d_%s_%d" % [exp_id, self.eval_use_heuristic, self.eval_beam_size],
    eval_output: "ie_dirs",
    eval_beam_size: 1,
    eval_use_heuristic: true,
    eval_steps: [ 1000 * x + 100 for x in std.range(0, 100)]+[101000],
    eval_section: "val",
}