// Domain Adaptation with bears using all chromosome for all populations
// Using ghost 2 simulations


process train {

    //memory { 1.GB * batch_size / 16 }

    input:
    tuple val(replicate), val(lambda), val(batch_size), val(learn_rate), val(enc_learn_rate), val(disc_learn_rate)
    
    script:
    """
    /mnt/scratch/smithlab/megan/da_revision/scripts/train_sfs_bear_real_IWCDAN.py \
      --rep ${replicate} \
      --max_lambda ${lambda} \
      --batch ${batch_size} \
      --learn ${learn_rate} \
      --enc_learn ${enc_learn_rate} \
      --disc_learn ${disc_learn_rate} \
      --epochs 50 \
      --force \
      --importance_weights
    """
}

workflow {
  replicate = channel.of("01", "02", "03", "04", "05", "06", "07", "08", "09", "10")
  lambda = channel.of(0, 0.1, 0.5)
  batch_size = channel.of(16)
  learn_rate = channel.of(1e-3)
  enc_learn = channel.of(1e-3)
  disc_learn = channel.of(1e-3)
  comb = replicate.combine(lambda).combine(batch_size).combine(learn_rate).combine(enc_learn).combine(disc_learn)
  train(comb)
}


