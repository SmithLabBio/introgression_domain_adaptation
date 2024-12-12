set sim_dir /mnt/home/kc2824/fscratch/popai/simulations/ 

echo "General Source:"
./site_cnt.py SecondaryContact \
  $sim_dir/general-secondary-contact-1-20000-train.json \
  $sim_dir/general-secondary-contact-1-1000-train.json \
  $sim_dir/general-secondary-contact-1-1000-test.json

echo "General Target:" 
./site_cnt.py GhostSecondaryContact \
  $sim_dir/general-secondary-contact-ghost-1-100-train.json \
  $sim_dir/general-secondary-contact-ghost-1-1000-test.json

echo "Bear Source:" 
./site_cnt.py SecondaryContact \
  $sim_dir/simulations/bear-secondary-contact-2-20000-train.json \
  $sim_dir/simulations/bear-secondary-contact-2-1000-train.json \
  $sim_dir/simulations/bear-secondary-contact-2-1000-test.jso

echo "Bear Target:"
./site_cnt.py GhostSecondaryContact \
  $sim_dir/simulations/bear-secondary-contact-ghost-2-100-train.json \
  $sim_dir/simulations/bear-secondary-contact-ghost-2-1000-test.json

