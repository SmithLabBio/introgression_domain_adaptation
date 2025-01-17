#!/usr/bin/env fish

sub -n convert-sim-20000 -m 100 -p normal -w "./convert_to_numpy2.py NumpyAfsDataset SecondaryContact /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-2-20000-train.json          sfs-norm --force --normalized"
sub -n convert-sim-1000         -p normal -w "./convert_to_numpy2.py NumpyAfsDataset SecondaryContact /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-2-1000-test.json            sfs-norm --force --normalized"
sub -n convert-sim-1000         -p normal -w "./convert_to_numpy2.py NumpyAfsDataset SecondaryContact /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-2-1000-train.json           sfs-norm --force --normalized"
sub -n convert-sim-ghost-1000   -p normal -w "./convert_to_numpy2.py NumpyAfsDataset GhostSecondaryContact /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-ghost-2-1000-test.json sfs-norm --force --normalized"
sub -n convert-sim-ghost-100    -p normal -w "./convert_to_numpy2.py NumpyAfsDataset GhostSecondaryContact /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-ghost-2-100-train.json sfs-norm --force --normalized"
sub -n convert-sim-ghost-36     -p normal -w "./convert_to_numpy2.py NumpyAfsDataset GhostSecondaryContact /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-ghost-2-36-train.json  sfs-norm --force --normalized"