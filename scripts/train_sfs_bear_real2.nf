// Domain Adaptation with bears using all chromosome for each population 

populations = [ 
  "brown-abc",
  "brown-alaska",
  // "brown-asia",
  "brown-eurasia",
  // "brown-eu",
  // "brown-hudson",
  // "brown-scandanavia",
  // "brown-us"
  ]

process train {

    memory { 1.GB * batch_size / 16 }

    input:
    tuple val(populations), val(replicate), val(lambda), val(gamma), val(batch_size), val(learn_rate), val(enc_learn_rate), val(disc_learn_rate)
    
    script:
    """
    /mnt/home/kc2824/popAI/kerry_evo/networks/train_sfs_bear_real2.py \
      --pops ${populations} \
      --rep ${replicate} \
      --max_lambda ${lambda} \
      --gamma ${gamma} \
      --batch ${batch_size} \
      --learn ${learn_rate} \
      --enc_learn ${enc_learn_rate} \
      --disc_learn ${disc_learn_rate} \
      --epochs 50
    """
}

workflow {
  samples = Channel.fromList(populations) 
  populations = samples.combine(samples).filter{ it[0] < it[1] }.map{ a, b -> "${a}_${b}" }
  replicate = Channel.of("01", "02", "03", "04", "05", "06", "07", "08", "09", "10")
  // lambda = channel.of(0.5, 1, 2, 10)
  lambda = Channel.of(0.1, 1, 10)
  gamma = Channel.of(1, 4, 10)
  // batch_size = channel.of(16, 64, 256)
  batch_size = Channel.of(16)
  learn_rate = Channel.of(1e-3)
  enc_learn = Channel.of(1e-3)
  disc_learn = Channel.of(1e-3)
  comb = populations.combine(replicate).combine(lambda).combine(gamma).combine(batch_size).combine(learn_rate).combine(enc_learn).combine(disc_learn)
  train(comb)
}


