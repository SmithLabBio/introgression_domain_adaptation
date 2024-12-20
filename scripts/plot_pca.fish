## Code to plot pca replicates individually
# set dir "/mnt/scratch/smithfs/cobb/popai/general1/"
# set params "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0"
# for i in 01 02 03 04 05 06 07 08 09 10 
#   ./plot_pca.py \
#     $dir$params"/$i/test-epoch-50/latent-space.npz" \
#     $dir"general1."$params".rep$i"
# end

# set params "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_1"
# for i in 01 02 03 04 05 06 07 08 09 10 
#   ./plot_pca.py \
#     $dir$params"/$i/test-epoch-50/latent-space.npz" \
#     $dir"general1."$params".rep$i"
# end

set dir "/mnt/scratch/smithfs/cobb/popai/general1/"
set params "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0"
./plot_pca.py \
  $dir$params \
  $dir"general1."$params

set params "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_1"
./plot_pca.py \
  $dir$params \
  $dir"general1."$params


set dir "/mnt/scratch/smithfs/cobb/popai/bear4/"
set params "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0"
./plot_pca.py \
  $dir$params \
  $dir"bear4."$params

set params "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0.5"
./plot_pca.py \
  $dir$params \
  $dir"bear4."$params


set dir "/mnt/scratch/smithfs/cobb/popai/bear7/"
set params "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0"
./plot_pca.py \
  $dir$params \
  $dir"bear7."$params

set params "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0.5"
./plot_pca.py \
  $dir$params \
  $dir"bear7."$params