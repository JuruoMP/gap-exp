{
    local exp_id = 1,
    logdir: "logdir/sparc_range_bart_run_%d" %exp_id,
    model_config: "configs/sparc_gap_range/gap-bart.jsonnet",
    model_config_args: {
        bs: 12,
        num_batch_accumulated: 2,
        bart_version: "facebook/bart-large",
        summarize_header: "avg",
        use_column_type: false,
        num_layers: 8,
        lr: 1e-4,
        bert_lr: 1e-5,
        att: 1,
        end_lr: 0,
        sc_link: true,
        cv_link: true,
        use_align_mat: true,
        use_align_loss: true,
        use_tc_loss: false,
        bart_token_type: true,
        decoder_hidden_size: 512,
        end_with_from: true, # equivalent to "SWGOIF" if true
        clause_order: null, # strings like "SWGOIF", it will be prioriotized over end_with_from
    },

    eval_name: "sparc_range_bart_run_%d_%s_%d" % [exp_id, self.eval_use_heuristic, self.eval_beam_size],
    eval_output: "ie_dirs",
    eval_beam_size: 1,
    eval_use_heuristic: true,
    eval_steps: [40500],#10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000,27200,28200,29200,30500,31500,32500,33500,34500,35500,36500,37500,38500,39500,
    eval_section: "val",
}
