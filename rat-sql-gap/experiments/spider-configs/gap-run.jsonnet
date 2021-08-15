{
    local exp_id = 1,
    logdir: "logdir/bart_run_%d" %exp_id,
    model_config: "configs/gap/gap-bart.jsonnet",
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
        bart_token_type: true,
        decoder_hidden_size: 512,
        end_with_from: true, # equivalent to "SWGOIF" if true
        clause_order: null, # strings like "SWGOIF", it will be prioriotized over end_with_from 
    },

    eval_name: "bart_run_%d_%s_%d" % [exp_id, self.eval_use_heuristic, self.eval_beam_size],
    eval_output: "ie_dirs",
    eval_beam_size: 1,
    eval_use_heuristic: true,
    eval_steps: [100,1100,2100,3100,4100,5100,6100,7100,8100,9100,10100,11100,12100,13100,14100,15100,16100,17100,18100,19100,20100,21100,22100,23100,24100,25100,26100,27100,28100,29100,30100,31100,32100,33100,34100,35100,36100,37100,38100,39100,40100,41000],
    eval_section: "val",
}