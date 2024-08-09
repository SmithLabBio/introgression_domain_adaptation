
process train {

    memory { 
      if (method == 'sfs') { return 1.GB * batch_size / 8 } 
      else if (method == 'norm') { return 2.GB * batch_size / 8 } 
    }

    input:
    tuple val(method), val(format), val(replicate), val(batch_size), val(opt_learn_rate), val(enc_learn_rate)
    
    script:
    """
    /mnt/home/kc2824/popAI/kerry_evo/networks/train_sfs_bear3.py \
      --meth ${method} \
      --format ${format} \
      --rep ${replicate} \
      --batch ${batch_size} \
      --learn ${opt_learn_rate} \
      --enc_learn ${enc_learn_rate} \
      --epochs 50
    """
}

workflow {
    method = channel.of("finetune", "cdan")
    // format = channel.of("sfs", "norm")
    format = channel.of("sfs")
    replicate = channel.of("01", "02", "03", "04", "05", "06", "07", "08", "09", "10")
    // batch_size = channel.of(16, 64, 256)
    batch_size = channel.of(256)
    // opt_learn_rate = channel.of(1e-3, 1e-4, 1e-6)
    opt_learn_rate = channel.of(1e-4)
    // enc_learn_rate = channel.of(1e-3, 1e-4, 1e-6) 
    enc_learn_rate = channel.of(1e-4) 
    comb = method.combine(format).combine(replicate).combine(batch_size).combine(opt_learn_rate).combine(enc_learn_rate)
    train(comb)
}


