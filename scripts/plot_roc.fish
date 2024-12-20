
## ROC curves plotted together
# set dir "/mnt/home/kc2824/fscratch/popai/general1" 
# ./roc_curves.py \
#   "General" \
#   $dir/"batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0/" \
#   $dir/"batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_1/" \
#   "test-epoch-50" \
#   $dir/"general1.batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda0-1.norm-afs.roc.pdf"

# set dir "/mnt/home/kc2824/fscratch/popai/bear4" 
# ./roc_curves.py \
#   "Bear" \
#   $dir/"batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0/" \
#   $dir/"batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0.5/" \
#   "test-epoch-50" \
#   $dir/"bear4.batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda0-0.5.norm-afs.roc.pdf"

# set dir "/mnt/home/kc2824/fscratch/popai/bear7" 
# ./roc_curves.py \
#   "Bear" \
#   $dir/"batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0/" \
#   $dir/"batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0.5/" \
#   "test-epoch-50" \
#   $dir/"bear7.batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda0-0.5.norm-afs.roc.pdf"


## ROC curves plotted separately

# general1
set dir "/mnt/home/kc2824/fscratch/popai/general1" 

set param "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0"
echo $param
set outpath $dir"/general1."$param".roc.pdf"
./plot_roc.py \
  $dir/$param \
  "test-epoch-50" \
  $outpath
echo $outpath

set param "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_1"
echo $param
set outpath $dir"/general1."$param".roc.pdf"
./plot_roc.py \
  $dir/$param \
  "test-epoch-50" \
  $outpath
echo $outpath


# bear1
set dir "/mnt/home/kc2824/fscratch/popai/bear4" 

set param "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0"
echo $param
set outpath  $dir"/bear4."$param".roc.pdf"
./plot_roc.py \
  $dir/$param \
  "test-epoch-50" \
  $outpath
echo $outpath

set param "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0.5"
echo $param
set outpath $dir"/bear4."$param".roc.pdf"
./plot_roc.py \
  $dir/$param \
  "test-epoch-50" \
  $outpath
echo $outpath


# bear2 
set dir "/mnt/home/kc2824/fscratch/popai/bear7" 

set param "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0"
echo $param
set outpath $dir"/bear7."$param".roc.pdf"
./plot_roc.py \
  $dir/$param \
  "test-epoch-50" \
  $outpath
echo $outpath

set param "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0.5"
echo $param
set outpath $dir"/bear7."$param".roc.pdf"
./plot_roc.py \
  $dir/$param \
  "test-epoch-50" \
  $outpath
echo $outpath