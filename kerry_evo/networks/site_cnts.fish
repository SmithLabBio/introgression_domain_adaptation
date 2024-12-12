echo "General Source:"
./site_cnt.py SecondaryContact \
  /mnt/home/kc2824/fscratch/popai/simulations/general-secondary-contact-1-20000-train.json \
  /mnt/home/kc2824/fscratch/popai/simulations/general-secondary-contact-1-1000-train.json  # Validation

echo "General Target:"
./site_cnt.py GhostSecondaryContact \ 
  /mnt/home/kc2824/fscratch/popai/simulations/general-secondary-contact-ghost-1-100-train.json

echo "Bear Source:"
./site_cnt.py SecondaryContact \
  /mnt/home/kc2824/fscratch/popai/simulations/bear-secondary-contact-2-20000-train.json \
  /mnt/home/kc2824/fscratch/popai/simulations/bear-secondary-contact-2-1000-train.json  # Validation

echo "Bear Target:"
./site_cnt.py GhostSecondaryContact \ 
  /mnt/home/kc2824/fscratch/popai/simulations/bear-secondary-contact-ghost-2-100-train.json