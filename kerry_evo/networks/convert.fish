#!/usr/bin/env fish

set n_snps 1500
set sim_dir "/mnt/scratch/smithfs/cobb/popai/simulations/"

# for i in (ls $sim_dir/general-secondary-contact-1*.json)  
#   sub -n convert -m 16 -w "./convert_to_numpy.py NumpyAfsDataset SecondaryContact $i afs --force"
#   sleep .1
#   sub -n convert -m 16 -w "./convert_to_numpy.py NumpySnpDataset SecondaryContact $i snps --n_snps $n_snps --force --polarized"
#   sleep .1
# end

# for i in (ls $sim_dir/general-secondary-contact-ghost-1*.json)
#   sub -n convert -m 16 -w "./convert_to_numpy.py NumpyAfsDataset GhostSecondaryContact $i afs --force"
#   sleep .1
#   sub -n convert -m 16 -w "./convert_to_numpy.py NumpySnpDataset GhostSecondaryContact $i snps --n_snps $n_snps --force --polarized"
#   sleep .1
# end


for i in (ls $sim_dir/bear-secondary-contact-1*.json)  
  sub -n convert -m 16 -w "./convert_to_numpy.py NumpyAfsDataset SecondaryContact $i afs --force"
  sleep .1
  # sub -n convert -m 16 -w "./convert_to_numpy.py NumpySnpDataset SecondaryContact $i snps --n_snps $n_snps --force"
  # sleep .1
end

for i in (ls $sim_dir/bear-secondary-contact-ghost-1*.json)
  sub -n convert -m 16 -w "./convert_to_numpy.py NumpyAfsDataset GhostSecondaryContact $i afs --force"
  sleep .1
  # sub -n convert -m 16 -w "./convert_to_numpy.py NumpySnpDataset GhostSecondaryContact $i snps --n_snps $n_snps --force"
  # sleep .1
end