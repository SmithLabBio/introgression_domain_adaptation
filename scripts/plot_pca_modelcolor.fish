set outdir "/mnt/scratch/smithlab/megan/da_revision/plotting_sfs/"

#set dir "/mnt/scratch/smithfs/cobb/popai/general1/"
#set params "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0"
#../scripts/plot_pca_modelcolor.py \
#  $dir$params \
#  $outdir"general1."$params
#
#set params "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_1"
#../scripts/plot_pca_modelcolor.py \
#  $dir$params \
#  $outdir"general1."$params



set dir "/mnt/scratch/smithfs/cobb/popai/bear7/"
set params "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0"
../scripts/plot_pca_modelcolor.py \
  $dir$params \
  $outdir"bear7."$params

set params "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0.5"
../scripts/plot_pca_modelcolor.py \
  $dir$params \
  $outdir"bear7."$params