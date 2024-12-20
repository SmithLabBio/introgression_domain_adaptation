
params.sim_dir = "/mnt/scratch/smithfs/cobb/popai/simulations/"
params.out_base = "/mnt/scratch/smithfs/cobb/popai/bear"
// params.target_data = "/mnt/scratch/smithlab/cobb/bears/filtered/numpy-dataset.npz"
// params.out_prefix = "real-"

params.target_data = "/mnt/scratch/smithlab/cobb/bears/filtered/numpy-dataset-NW_026623050.1.npz"
params.out_prefix = "real-NW_026623050.1-"

process train {
    maxRetries 2 
    errorStrategy { (task.attempt <= maxRetries) ? "retry" : "ignore" }
    memory { data_type == "afs" ? 4.GB * task.attempt : 32.GB * task.attempt }

    input:
    tuple val(data_type), val(max_lambda), val(rate_ratio), val(replicate)
    
    script:
    """
    /mnt/home/kc2824/popAI/kerry_evo/networks/train.py \
      --ModelFile model1 \
      --data_type ${data_type} \
      --source_path ${params.sim_dir}/bear-secondary-contact-1-20000-train-${data_type}.npz \
      --target_path ${params.target_data} \
      --val_path    None \
      --max_lambda ${max_lambda} \
      --learn_rate 0.0001 \
      --disc_enc_learn_ratio ${rate_ratio} \
      --outdir ${params.out_base}/${params.out_prefix}model1-${data_type}-max_lambda_${max_lambda}-rate_ratio_${rate_ratio}-rate_1e-4/${replicate} \
      --force
    """
}

workflow {
    // data = channel.of("snp")
    data = channel.of("afs")
    replicate = channel.of("01", "02", "03", "04", "05", "06", "07", "08", "09", "10")
    max_lambda = channel.of(0.1)
    rate_ratio = channel.of(1) 
    cdan = data.combine(max_lambda).combine(rate_ratio).combine(replicate)
    // Create replicates with max_lambda and rate_ratio of zero for orig training
    // orig = data.combine(replicate).map{a, b -> tuple(a, 0, 0, b)}
    // Combine all
    // all = cdan.concat(orig)
    // train(all)
    train(cdan)
}

