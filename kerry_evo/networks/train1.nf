
params.sim_dir = "/mnt/scratch/smithfs/cobb/popai/simulations/"
params.out_base = "/mnt/scratch/smithfs/cobb/popai/output2"

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
      --source_path ${params.sim_dir}/general-secondary-contact-1-20000-train-${data_type}.npz \
      --target_path ${params.sim_dir}/general-secondary-contact-ghost-1-100-train-${data_type}.npz \
      --val_path    ${params.sim_dir}/general-secondary-contact-ghost-1-100-val-${data_type}.npz \
      --max_lambda ${max_lambda} \
      --learn_rate 0.0001 \
      --disc_enc_learn_ratio ${rate_ratio} \
      --outdir ${params.out_base}/model1-${data_type}-max_lambda_${max_lambda}-rate_ratio_${rate_ratio}-rate_1e-4/${replicate} \
      --force
    """
}

workflow {
    data = channel.of("snp")
    // data = channel.of("afs")
    replicate = channel.of("01", "02", "03", "04", "05", "06", "07", "08", "09", "10")
    max_lambda = channel.of(1, 10, 20)
    rate_ratio = channel.of(0.1, 1, 10, 20) 
    cdan = data.combine(max_lambda).combine(rate_ratio).combine(replicate)
    // Create replicates with max_lambda and rate_ratio of zero for orig training
    orig = data.combine(replicate).map{a, b -> tuple(a, 0, 0, b)}
    // Combine all
    all = cdan.concat(orig)
    train(all)
}




// # # AFS CDAN  
// # set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
// # set out_base "/mnt/scratch/smithfs/cobb/popai/output"
// # set n_snps 1500
// # for i in 0.1 1 5 10 20
// #   set name "model1-rate_ratio_$i-rate_1e-4"
// #   for rep in 01 02 03 04 05 06 07 08 09 10
// #     set cmd "./train.py \
// #       --ModelFile model1 \
// #       --DataType NumpyAfsDataset \
// #       --SrcType SecondaryContact \
// #       --TgtType GhostSecondaryContact \
// #       --source_path $sim_dir/general-secondary-contact-1-1000-train.json \
// #       --target_path $sim_dir/general-secondary-contact-ghost-1-100-train.json \
// #       --val_path    $sim_dir/general-secondary-contact-ghost-1-100-val.json \
// #       --n_snps $n_snps \
// #       --max_lambda 10 \
// #       --learn_rate 0.0001 \
// #       --disc_enc_learn_ratio $i \
// #       --outdir $out_base/afs-cdan/$name/$rep"
// #     sub -n train-afs-cdan-$name-$rep -o logs -m 8 -w $cmd
// #     end
// # end


// # # SNP Alignment with no domain adaptiation
// # set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
// # set out_base "/mnt/scratch/smithfs/cobb/popai/output"
// # set n_snps 1500
// # set name "model1-rate_1e-4"
// # for rep in 01 02 03 04 05 06 07 08 09 10
// #   set snp_cmd "./train.py \
// #     --ModelFile model1 \
// #     --DataType NumpySnpDataset \
// #     --SrcType SecondaryContact \
// #     --TgtType SecondaryContact \
// #     --source_path $sim_dir/general-secondary-contact-1-1000-train.json \
// #     --target_path $sim_dir/general-secondary-contact-1-100-train.json \
// #     --val_path    $sim_dir/general-secondary-contact-1-100-val.json \
// #     --n_snps $n_snps \
// #     --max_lambda 0 \
// #     --learn_rate 0.0001 \
// #     --disc_enc_learn_ratio 1 \
// #     --outdir $out_base/snp-orig/$name-$rep/"
// #   sub -n train-snp-orig-$name-$rep -o logs -m 8 -w $snp_cmd
// # end


// # # SNP CDAN
// # set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
// # set out_base "/mnt/scratch/smithfs/cobb/popai/output"
// # set n_snps 1500
// # for i in 0.1 1 5 10 20
// #   set name "model1-rate_ratio_$i-rate_1e-4" 
// #   for rep in 01 02 03 04 05 06 07 08 09 10
// #     set cmd "./train.py \
// #       --ModelFile model1 \
// #       --DataType NumpySnpDataset \
// #       --SrcType SecondaryContact \
// #       --TgtType GhostSecondaryContact \
// #       --source_path $sim_dir/general-secondary-contact-1-1000-train.json \
// #       --target_path $sim_dir/general-secondary-contact-ghost-1-100-train.json \
// #       --val_path    $sim_dir/general-secondary-contact-ghost-1-100-val.json \
// #       --n_snps $n_snps \
// #       --max_lambda 10 \
// #       --learn_rate 0.0001 \
// #       --disc_enc_learn_ratio $i \
// #       --outdir $out_base/snp-cdan/$name/$rep"
// #     sub -n train-snp-cdan-$name-$rep -o logs -m 8 -w $cmd
// #   end
// # end


// # # SNP CDAN lower learning rate
// # set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
// # set out_base "/mnt/scratch/smithfs/cobb/popai/output"
// # set n_snps 1500
// # for i in 0.1 1 5 10 20
// #   set name "model1-rate_ratio_$i-rate_1e-5" 
// #   for rep in 01 02 03 04 05 06 07 08 09 10
// #     set cmd "./train.py \
// #       --ModelFile model1 \
// #       --DataType NumpySnpDataset \
// #       --SrcType SecondaryContact \
// #       --TgtType GhostSecondaryContact \
// #       --source_path $sim_dir/general-secondary-contact-1-1000-train.json \
// #       --target_path $sim_dir/general-secondary-contact-ghost-1-100-train.json \
// #       --val_path    $sim_dir/general-secondary-contact-ghost-1-100-val.json \
// #       --n_snps $n_snps \
// #       --max_lambda 10 \
// #       --learn_rate 0.00001 \
// #       --disc_enc_learn_ratio $i \
// #       --outdir $out_base/snp-cdan/$name/$rep"
// #     sub -n train-snp-cdan-$name-$rep -o logs -m 8 -w $cmd
// #   end
// # end

// # # SNP CDAN lower lambda
// # set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
// # set out_base "/mnt/scratch/smithfs/cobb/popai/output"
// # set n_snps 1500
// # for i in 0.1 1 5 10 20
// #   set name "model1-rate_ratio_$i-rate_1e-4-lambda_1" 
// #   for rep in 01 02 03 04 05 06 07 08 09 10
// #     set cmd "./train.py \
// #       --ModelFile model1 \
// #       --DataType NumpySnpDataset \
// #       --SrcType SecondaryContact \
// #       --TgtType GhostSecondaryContact \
// #       --source_path $sim_dir/general-secondary-contact-1-1000-train.json \
// #       --target_path $sim_dir/general-secondary-contact-ghost-1-100-train.json \
// #       --val_path    $sim_dir/general-secondary-contact-ghost-1-100-val.json \
// #       --n_snps $n_snps \
// #       --max_lambda 1 \
// #       --learn_rate 0.0001 \
// #       --disc_enc_learn_ratio $i \
// #       --outdir $out_base/snp-cdan/$name/$rep"
// #     sub -n train-snp-cdan-$name-$rep -o logs -m 8 -w $cmd
// #   end
// # end


// # # SNP CDAN lower learning rate and lower lambda
// # set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
// # set out_base "/mnt/scratch/smithfs/cobb/popai/output"
// # set n_snps 1500
// # for i in 0.1 1 5 10 20
// #   set name "model1-rate_ratio_$i-rate_1e-5-lambda_1" 
// #   for rep in 01 02 03 04 05 06 07 08 09 10
// #     set cmd "./train.py \
// #       --ModelFile model1 \
// #       --DataType NumpySnpDataset \
// #       --SrcType SecondaryContact \
// #       --TgtType GhostSecondaryContact \
// #       --source_path $sim_dir/general-secondary-contact-1-1000-train.json \
// #       --target_path $sim_dir/general-secondary-contact-ghost-1-100-train.json \
// #       --val_path    $sim_dir/general-secondary-contact-ghost-1-100-val.json \
// #       --n_snps $n_snps \
// #       --max_lambda 1 \
// #       --learn_rate 0.00001 \
// #       --disc_enc_learn_ratio $i \
// #       --outdir $out_base/snp-cdan/$name/$rep"
// #     sub -n train-snp-cdan-$name-$rep -o logs -m 8 -w $cmd
// #   end
// # end

// # # SNP CDAN with fixed lambda 10
// # set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
// # set out_base "/mnt/scratch/smithfs/cobb/popai/output"
// # set n_snps 1500
// # for i in 0.1 1 5 10 20
// #   set name "model1-rate_ratio_$i-rate_1e-4-fix_lambda_10" 
// #   for rep in 01 02 03 04 05 06 07 08 09 10
// #     set cmd "./train.py \
// #       --ModelFile model1 \
// #       --DataType NumpySnpDataset \
// #       --SrcType SecondaryContact \
// #       --TgtType GhostSecondaryContact \
// #       --source_path $sim_dir/general-secondary-contact-1-1000-train.json \
// #       --target_path $sim_dir/general-secondary-contact-ghost-1-100-train.json \
// #       --val_path    $sim_dir/general-secondary-contact-ghost-1-100-val.json \
// #       --n_snps $n_snps \
// #       --max_lambda 10 \
// #       --learn_rate 0.0001 \
// #       --disc_enc_learn_ratio $i \
// #       --outdir $out_base/snp-cdan/$name/$rep \
// #       --static_lambda"
// #     sub -n train-snp-cdan-$name-$rep -o logs -m 8 -w $cmd
// #   end
// # end

// # # SNP CDAN with fixed lambda 1
// # set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
// # set out_base "/mnt/scratch/smithfs/cobb/popai/output"
// # set n_snps 1500
// # for i in 0.1 1 5 10 20
// #   set name "model1-rate_ratio_$i-rate_1e-4-fix_lambda_1" 
// #   for rep in 01 02 03 04 05 06 07 08 09 10
// #     set cmd "./train.py \
// #       --ModelFile model1 \
// #       --DataType NumpySnpDataset \
// #       --SrcType SecondaryContact \
// #       --TgtType GhostSecondaryContact \
// #       --source_path $sim_dir/general-secondary-contact-1-1000-train.json \
// #       --target_path $sim_dir/general-secondary-contact-ghost-1-100-train.json \
// #       --val_path    $sim_dir/general-secondary-contact-ghost-1-100-val.json \
// #       --n_snps $n_snps \
// #       --max_lambda 1 \
// #       --learn_rate 0.0001 \
// #       --disc_enc_learn_ratio $i \
// #       --outdir $out_base/snp-cdan/$name/$rep \
// #       --static_lambda"
// #     sub -n train-snp-cdan-$name-$rep -o logs -m 8 -w $cmd
// #   end
// # end


// # # SNP CDAN with fixed lambda 10
// # set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
// # set out_base "/mnt/scratch/smithfs/cobb/popai/output"
// # set n_snps 1500
// # for i in 0.1 1 5 10 20
// #   set name "model1-rate_ratio_$i-rate_1e-4-fix_lambda_.1" 
// #   for rep in 01 02 03 04 05 06 07 08 09 10
// #     set cmd "./train.py \
// #       --ModelFile model1 \
// #       --DataType NumpySnpDataset \
// #       --SrcType SecondaryContact \
// #       --TgtType GhostSecondaryContact \
// #       --source_path $sim_dir/general-secondary-contact-1-1000-train.json \
// #       --target_path $sim_dir/general-secondary-contact-ghost-1-100-train.json \
// #       --val_path    $sim_dir/general-secondary-contact-ghost-1-100-val.json \
// #       --n_snps $n_snps \
// #       --max_lambda .1 \
// #       --learn_rate 0.0001 \
// #       --disc_enc_learn_ratio $i \
// #       --outdir $out_base/snp-cdan/$name/$rep \
// #       --static_lambda"
// #     sub -n train-snp-cdan-$name-$rep -o logs -m 8 -w $cmd
// #   end
// # end


// # AFS with no domain adaptation
// set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
// set out_base "/mnt/scratch/smithfs/cobb/popai/output2"
// set n_snps 1500
// set name "model1-rate_1e-4"
// for rep in 01 02 03 04 05 06 07 08 09 10
//   set cmd "./train.py \
//     --ModelFile model1 \
//     --DataType NumpyAfsDataset \
//     --SrcType SecondaryContact \
//     --TgtType SecondaryContact \
//     --source_path $sim_dir/general-secondary-contact-1-20000-train.json \
//     --target_path $sim_dir/general-secondary-contact-1-100-train.json \
//     --val_path    $sim_dir/general-secondary-contact-1-100-val.json \
//     --n_snps $n_snps \
//     --max_lambda 0 \
//     --learn_rate 0.0001 \
//     --disc_enc_learn_ratio 1 \
//     --outdir $out_base/afs-orig/$name-$rep/"
//   sub -n train-afs-orig-$name-$rep -o logs -m 8 -w $cmd
//   sleep 1
// end


// for i in 0.1 1 5 10 20
// # AFS CDAN  
// set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
// set out_base "/mnt/scratch/smithfs/cobb/popai/output2"
// set n_snps 1500
// for i in 1
//   set name "model1-rate_ratio_$i-rate_1e-4"
//   for rep in 01 02 03 04 05 06 07 08 09 10
//     set cmd "./train.py \
//       --ModelFile model1 \
//       --DataType NumpyAfsDataset \
//       --SrcType SecondaryContact \
//       --TgtType GhostSecondaryContact \
//       --source_path $sim_dir/general-secondary-contact-1-20000-train.json \
//       --target_path $sim_dir/general-secondary-contact-ghost-1-100-train.json \
//       --val_path    $sim_dir/general-secondary-contact-ghost-1-100-val.json \
//       --n_snps $n_snps \
//       --max_lambda 10 \
//       --learn_rate 0.0001 \
//       --disc_enc_learn_ratio $i \
//       --outdir $out_base/afs-cdan/$name/$rep"
//     sub -n train-afs-cdan-$name-$rep -o logs -m 8 -w $cmd
//     end
// end