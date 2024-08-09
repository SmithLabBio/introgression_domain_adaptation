// Compare normalized sfs to non-normalized

process train {

    memory { 
      if (format == 'sfs') { return 1.GB * batch_size / 16 } 
      else if (format == 'norm') { return 2.GB * batch_size / 16 } 
    }

    input:
    tuple val(format), val(replicate), val(batch_size), val(learn_rate)
    
    script:
    """
    /mnt/home/kc2824/popAI/kerry_evo/networks/train_sfs_bear2.py \
      --format ${format} \
      --rep ${replicate} \
      --batch ${batch_size} \
      --learn ${learn_rate} \
      --enc_learn ${learn_rate} \
      --epochs 100
    """
}

workflow {
    format = channel.of("sfs", "norm")
    replicate = channel.of("01", "02", "03", "04", "05", "06", "07", "08", "09", "10")
    batch_size = channel.of(16, 64, 256)
    learn_rate = channel.of(1e-3, 1e-4, 1e-5)
    comb = format.combine(replicate).combine(batch_size).combine(learn_rate)
    train(comb)
}


