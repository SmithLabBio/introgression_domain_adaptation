# training IWCDAN
sbatch --partition normal --output=slurm-%j.out --error=slurm-%j.err --wrap "/mnt/home/ms4438/programs/nextflow -C ./scripts/train.config run ./scripts/train_sfs_adjustable_iwcdan.nf -resume"
sbatch --partition normal --output=slurm-%j.out --error=slurm-%j.err --wrap "/mnt/home/ms4438/programs/nextflow -C ./scripts/train.config run ./scripts/train_sfs_adjustable_noiwcdan.nf -resume"

# testing IWCDAN
sbatch -p normal --wrap "/mnt/home/ms4438/programs/nextflow -C ../scripts/test.config run ../scripts/test_sfs_IWCDAN.nf"
sbatch -p normal --wrap "/mnt/home/ms4438/programs/nextflow -C ../scripts/test.config run ../scripts/test_sfs_noIWCDAN.nf"

# plotting IWCDAN


# training IWCDAN for bear
sbatch --partition normal --output=slurm-%j.out --error=slurm-%j.err --wrap "/mnt/home/ms4438/programs/nextflow -C ./scripts/train.config run ./scripts/train_sfs_adjustable_iwcdan_bear.nf -resume"
sbatch --partition normal --output=slurm-%j.out --error=slurm-%j.err --wrap "/mnt/home/ms4438/programs/nextflow -C ./scripts/train.config run ./scripts/train_sfs_adjustable_noiwcdan_bear.nf -resume"

# testing IWCDAN for bear
sbatch -p normal --wrap "/mnt/home/ms4438/programs/nextflow -C ../scripts/test.config run ../scripts/test_sfs_IWCDAN_bear.nf"
sbatch -p normal --wrap "/mnt/home/ms4438/programs/nextflow -C ../scripts/test.config run ../scripts/test_sfs_noIWCDAN_bear.nf"

# empirical bears
## first combine SFS
python scripts/combine_empirical_bear_npy.py -i /mnt/scratch/smithlab/cobb/bears/filtered/ -o empirical_bears_iwcdan/combined_sfs.npz
sbatch --partition smith --output=slurm-%j.out --error=slurm-%j.err --wrap "/mnt/home/ms4438/programs/nextflow -C ./scripts/train.config run ./scripts/train_sfs_bear_real_IWCDAN.nf -resume"
