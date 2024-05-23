# SNP Orig 
set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
set out_base "/mnt/scratch/smithfs/cobb/popai/output"
set name "model1-rate_1e-4"
for rep in 01 02 03 04 05 06 07 08 09 10
  set snp_cmd "./test.py \
    --json_path $out_base/snp-orig/$name-$rep/config.json \
    --source_path $sim_dir/general-secondary-contact-1-100-test.json \
    --target_path $sim_dir/general-secondary-contact-ghost-1-100-test.json \
    --epoch 50"
  sub -n test-snp-orig-$name-$rep -o logs -m 4 -w $snp_cmd
end


# AFS Orig 
set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
set out_base "/mnt/scratch/smithfs/cobb/popai/output"
set name "model1-rate_1e-4"
for rep in 01 02 03 04 05 06 07 08 09 10
  set cmd "./test.py \
    --json_path $out_base/afs-orig/$name-$rep/config.json \
    --source_path $sim_dir/general-secondary-contact-1-100-test.json \
    --target_path $sim_dir/general-secondary-contact-ghost-1-100-test.json \
    --epoch 50"
  sub -n test-afs-orig-$name-$rep -o logs -m 4 -w $cmd
end


# SNP CDAN 
set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
set out_base "/mnt/scratch/smithfs/cobb/popai/output"
for i in 0.1 1 5 10
  set name "model1-rate_ratio_$i-rate_1e-4" 
  for rep in 01 02 03 04 05 06 07 08 09 10
    set cmd "./test.py \
      --json_path $out_base/snp-cdan/$name/$rep/config.json \
      --source_path $sim_dir/general-secondary-contact-1-100-test.json \
      --target_path $sim_dir/general-secondary-contact-ghost-1-100-test.json \
      --epoch 50"
    sub -n test-snp-cdan-$name-$rep -o logs -m 4 -w $cmd
  end
end

# SNP CDAN 
set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
set out_base "/mnt/scratch/smithfs/cobb/popai/output"
for i in 0.1 1 5 10
  set name "model1-rate_ratio_$i-rate_1e-5" 
  for rep in 01 02 03 04 05 06 07 08 09 10
    set cmd "./test.py \
      --json_path $out_base/snp-cdan/$name/$rep/config.json \
      --source_path $sim_dir/general-secondary-contact-1-100-test.json \
      --target_path $sim_dir/general-secondary-contact-ghost-1-100-test.json \
      --epoch 50"
    sub -n test-snp-cdan-$name-$rep -o logs -m 4 -w $cmd
  end
end


# AFS CDAN
set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
set out_base "/mnt/scratch/smithfs/cobb/popai/output"
for i in 0.1 1 5 10
  set name "model1-rate_ratio_$i-rate_1e-4" 
  for rep in 01 02 03 04 05 06 07 08 09 10
    set cmd "./test.py \
      --json_path $out_base/afs-cdan/$name/$rep/config.json \
      --source_path $sim_dir/general-secondary-contact-1-100-test.json \
      --target_path $sim_dir/general-secondary-contact-ghost-1-100-test.json \
      --epoch 50"
    sub -n test-afs-cdan-$name-$rep -o logs -m 4 -w $cmd
  end
end