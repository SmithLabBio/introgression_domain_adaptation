
# # SNP Alignment with no mispec
# set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
# set out_base "/mnt/scratch/smithfs/cobb/popai/output"
# set n_snps 1500
# set name "model1-rate_1e-4"
# for rep in 01 02 03 04 05 06 07 08 09 10
#   set snp_cmd "./train.py \
#     --ModelFile model1 \
#     --DataType NumpySnpDataset \
#     --SrcType SecondaryContact \
#     --TgtType SecondaryContact \
#     --source_path $sim_dir/general-secondary-contact-1-1000-train.json \
#     --target_path $sim_dir/general-secondary-contact-1-100-train.json \
#     --val_path    $sim_dir/general-secondary-contact-1-100-val.json \
#     --n_snps $n_snps \
#     --max_lambda 0 \
#     --learn_rate 0.0001 \
#     --disc_enc_learn_ratio 1 \
#     --outdir $out_base/snp-orig/$name-$rep/"
#   sub -n train-snp-orig-$name-$rep -o logs -m 16 -w $snp_cmd
# end


# # AFS with no mispec
# set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
# set out_base "/mnt/scratch/smithfs/cobb/popai/output"
# set n_snps 1500
# set name "model1-rate_1e-4"
# for rep in 01 02 03 04 05 06 07 08 09 10
#   set cmd "./train.py \
#     --ModelFile model1 \
#     --DataType NumpyAfsDataset \
#     --SrcType SecondaryContact \
#     --TgtType SecondaryContact \
#     --source_path $sim_dir/general-secondary-contact-1-1000-train.json \
#     --target_path $sim_dir/general-secondary-contact-1-100-train.json \
#     --val_path    $sim_dir/general-secondary-contact-1-100-val.json \
#     --n_snps $n_snps \
#     --max_lambda 0 \
#     --learn_rate 0.0001 \
#     --disc_enc_learn_ratio 1 \
#     --outdir $out_base/afs-orig/$name-$rep/"
#   sub -n train-afs-orig-$name-$rep -o logs -m 16 -w $cmd
# end



# SNP mispec
set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
set out_base "/mnt/scratch/smithfs/cobb/popai/output"
set n_snps 1500
for i in 0.1 1 5 10
  set name "model1-rate_ratio_$i-rate_1e-4" 
  for rep in 01 02 03 04 05 06 07 08 09 10
    set cmd "./train.py \
      --ModelFile model1 \
      --DataType NumpySnpDataset \
      --SrcType SecondaryContact \
      --TgtType GhostSecondaryContact \
      --source_path $sim_dir/general-secondary-contact-1-1000-train.json \
      --target_path $sim_dir/general-secondary-contact-ghost-1-100-train.json \
      --val_path    $sim_dir/general-secondary-contact-ghost-1-100-val.json \
      --n_snps $n_snps \
      --max_lambda 10 \
      --learn_rate 0.0001 \
      --disc_enc_learn_ratio $i \
      --outdir $out_base/snp-mispec/$name/$rep"
    sub -n train-snp-$name-$rep -o logs -m 16 -w $cmd
  end
end



# AFS mispec
set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
set out_base "/mnt/scratch/smithfs/cobb/popai/output"
set n_snps 1500
for i in 0.1 1 5 10
  set name "model1-rate_ratio_$i-rate_1e-4"
  for rep in 01 02 03 04 05 06 07 08 09 10
    set cmd "./train.py \
      --ModelFile model1 \
      --DataType NumpyAfsDataset \
      --SrcType SecondaryContact \
      --TgtType GhostSecondaryContact \
      --source_path $sim_dir/general-secondary-contact-1-1000-train.json \
      --target_path $sim_dir/general-secondary-contact-ghost-1-100-train.json \
      --val_path    $sim_dir/general-secondary-contact-ghost-1-100-val.json \
      --n_snps $n_snps \
      --max_lambda 10 \
      --learn_rate 0.0001 \
      --disc_enc_learn_ratio $i \
      --outdir $out_base/afs-mispec/$name/$rep"
    sub -n train-afs-$name-$rep -o logs -m 16 -w $cmd
    end
end
