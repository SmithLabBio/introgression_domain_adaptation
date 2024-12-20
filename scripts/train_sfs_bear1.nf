// For enusuring that FineTuning and CDAN with lambda 0 produce equivalent results.

process train {

    memory 1.GB 

    input:
    tuple val(method), val(replicate), val(batch_size), val(opt_learn_rate), val(enc_learn_rate)
    
    script:
    """
    /mnt/home/kc2824/popAI/kerry_evo/networks/train_sfs_bear1.py \
      --meth ${method} \
      --rep ${replicate} \
      --batch ${batch_size} \
      --learn ${opt_learn_rate} \
      --enc_learn ${enc_learn_rate} \
      --epochs 50
    """
}

workflow {
    method = channel.of("finetune", "cdan")
    replicate = channel.of("01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20")
    batch_size = channel.of(256)
    opt_learn_rate = channel.of(1e-4)
    enc_learn_rate = channel.of(1e-4) 
    comb = method.combine(replicate).combine(batch_size).combine(opt_learn_rate).combine(enc_learn_rate)
    train(comb)
}


