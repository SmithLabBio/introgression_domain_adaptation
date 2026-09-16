set outdir "/mnt/scratch/smithlab/megan/da_revision/plotting_sfs/"

set dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
set src "general-secondary-contact-1-1000-test-sfs"
set tgt "general-secondary-contact-ghost-1-1000-test-sfs"
../scripts/plot_pcaofsfs.py \
  $dir$src \
  $dir$tgt \
  $outdir"general1."$src".vs."$tgt


set dir "/mnt/scratch/smithfs/cobb/popai/simulations/"
set src "bear-secondary-contact-2-1000-test-sfs"
set tgt "bear-secondary-contact-ghost-2-1000-test-sfs"
../scripts/plot_pcaofsfs.py \
  $dir$src \
  $dir$tgt \
  $outdir"bear."$src".vs."$tgt
