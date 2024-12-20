
// Domain adaptation with normalized trying multiple parameter combinations

process train {

    memory { 1.GB * batch_size / 16 }

    input:
    tuple val(replicate), val(lambda), val(batch_size), val(learn_rate), val(enc_learn_rate), val(disc_learn_rate)
    
    script:
    """
    /mnt/home/kc2824/popAI/kerry_evo/networks/train_sfs_bear4.py \
      --rep ${replicate} \
      --max_lambda ${lambda} \
      --batch ${batch_size} \
      --learn ${learn_rate} \
      --enc_learn ${enc_learn_rate} \
      --disc_learn ${disc_learn_rate} \
      --epochs 50
    """
}

workflow {
    replicate = channel.of("01", "02", "03", "04", "05", "06", "07", "08", "09", "10")
    // lambda = channel.of(0, 1, 2, 10)
    lambda = channel.of(0, 0.5)
    // batch_size = channel.of(32, 64, 256)
    batch_size = channel.of(16)
    // learn_rate = channel.of(1e-3, 1e-4)
    learn_rate = channel.of(1e-3)
    // enc_learn = channel.of(1e-3, 1e-4)
    enc_learn = channel.of(1e-3)
    // disc_learn = channel.of(1e-3, 1e-4)
    disc_learn = channel.of(1e-3)
    comb = replicate.combine(lambda).combine(batch_size).combine(learn_rate).combine(enc_learn).combine(disc_learn)
    train(comb)
}


