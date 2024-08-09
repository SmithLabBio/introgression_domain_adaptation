
params.sim_dir = "/mnt/scratch/smithfs/cobb/popai/simulations/"
params.out_base = "/mnt/scratch/smithfs/cobb/popai/bear"
params.target_dir = "/mnt/scratch/smithlab/cobb/bears/filtered/"

populations = [ 
  "brown-abc",
  "brown-alaska",
  "brown-asia",
  "brown-eurasia",
  "brown-eu",
  "brown-hudson",
  "brown-scandanavia",
  "brown-us"]

process train {
    // maxRetries 2 
    // errorStrategy { (task.attempt <= maxRetries) ? "retry" : "ignore" }
    // memory { data_type == "afs" ? 4.GB * task.attempt : 32.GB * task.attempt }

    input:
    tuple val(populations), val(data_type), val(max_lambda), val(rate_ratio), val(replicate)
    
    script:
    """
    /mnt/home/kc2824/popAI/kerry_evo/networks/train.py \
      --ModelFile model1 \
      --data_type ${data_type} \
      --source_path ${params.sim_dir}/bear-secondary-contact-1-20000-train-${data_type}.npz \
      --target_path ${params.target_dir}/${populations}.npz \
      --val_path    None \
      --max_lambda ${max_lambda} \
      --learn_rate 0.0001 \
      --disc_enc_learn_ratio ${rate_ratio} \
      --outdir ${params.out_base}/real-model1-${data_type}-max_lambda_${max_lambda}-rate_ratio_${rate_ratio}-rate_1e-4/${populations}/${replicate} \
      --force
    """
}

workflow {
    samples = Channel.fromList(populations) 
    comb = samples.combine(samples).filter{ it[0] < it[1] }.map{ a, b -> "${a}_${b}" }

    // data = channel.of("snp")
    data = channel.of("afs")
    replicate = channel.of("01", "02", "03", "04", "05", "06", "07", "08", "09", "10")
    max_lambda = channel.of(1, 10, 100)
    rate_ratio = channel.of(1, 10, 100) 
    cdan = comb.combine(data).combine(max_lambda).combine(rate_ratio).combine(replicate)

    train(cdan)
}

