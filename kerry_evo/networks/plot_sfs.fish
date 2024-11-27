echo gen-nomig
./plot_sfs.py ~/fscratch/popai/simulations/general-secondary-contact-1-1000-train-sfs-norm.npz ~/fscratch/popai/simulations/general-secondary-contact-1-1000-train-sfs-nomig.pdf --ix 0:499 --mean
echo gen-mig
./plot_sfs.py ~/fscratch/popai/simulations/general-secondary-contact-1-1000-train-sfs-norm.npz ~/fscratch/popai/simulations/general-secondary-contact-1-1000-train-sfs-mig.pdf --ix 500:-1 --mean

echo gen-nomig-ghost
./plot_sfs.py ~/fscratch/popai/simulations/general-secondary-contact-ghost-1-1000-train-sfs-norm.npz ~/fscratch/popai/simulations/general-secondary-contact-ghost-1-1000-train-sfs-nomig.pdf --ix 0:499 --mean
echo gen-mig-ghost
./plot_sfs.py ~/fscratch/popai/simulations/general-secondary-contact-ghost-1-1000-train-sfs-norm.npz ~/fscratch/popai/simulations/general-secondary-contact-ghost-1-1000-train-sfs-mig.pdf --ix 500:-1 --mean

echo bear-nomig
./plot_sfs.py ~/fscratch/popai/simulations/bear-secondary-contact-1-1000-train-sfs-norm.npz ~/fscratch/popai/simulations/bear-secondary-contact-1-1000-train-sfs-nomig.pdf --ix 0:499 --mean
echo bear-mig
./plot_sfs.py ~/fscratch/popai/simulations/bear-secondary-contact-1-1000-train-sfs-norm.npz ~/fscratch/popai/simulations/bear-secondary-contact-1-1000-train-sfs-mig.pdf --ix 500:-1 --mean

echo bear-nomig-ghost
./plot_sfs.py ~/fscratch/popai/simulations/bear-secondary-contact-ghost-1-1000-train-sfs-norm.npz ~/fscratch/popai/simulations/bear-secondary-contact-ghost-1-1000-train-sfs-nomig.pdf --ix 0:499 --mean
echo bear-mig-ghost
./plot_sfs.py ~/fscratch/popai/simulations/bear-secondary-contact-ghost-1-1000-train-sfs-norm.npz ~/fscratch/popai/simulations/bear-secondary-contact-ghost-1-1000-train-sfs-mig.pdf --ix 500:-1 --mean