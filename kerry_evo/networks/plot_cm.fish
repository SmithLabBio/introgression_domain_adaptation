
set dir "/mnt/scratch/smithfs/cobb/popai/general1/"

set prefix "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0"
./plot_cm.py \
  $dir$prefix \
  "test-epoch-50" \
  "$dir/general1.$prefix"

set prefix "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_1"
./plot_cm.py \
  $dir$prefix \
  "test-epoch-50" \
  "$dir/general1.$prefix"


# set dir "/mnt/scratch/smithfs/cobb/popai/bear4/"

# set prefix "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0"
# ./plot_cm.py \
#   $dir$prefix \
#   "test-epoch-50" \
#   "$dir/bear4.$prefix"

# set prefix "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0.5"
# ./plot_cm.py \
#   $dir$prefix \
#   "test-epoch-50" \
#   "$dir/bear4.$prefix"

set dir "/mnt/scratch/smithfs/cobb/popai/bear7/"

set prefix "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0"
./plot_cm.py \
  $dir$prefix \
  "test-epoch-50" \
  "$dir/bear7.$prefix"

set prefix "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0.5"
./plot_cm.py \
  $dir$prefix \
  "test-epoch-50" \
  "$dir/bear7.$prefix"
