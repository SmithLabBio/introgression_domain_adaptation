# training IWCDAN
sbatch --partition normal --output=slurm-%j.out --error=slurm-%j.err --wrap "/mnt/home/ms4438/programs/nextflow -C ./scripts/train.config run ./scripts/train_sfs_adjustable_iwcdan.nf -resume"
sbatch --partition normal --output=slurm-%j.out --error=slurm-%j.err --wrap "/mnt/home/ms4438/programs/nextflow -C ./scripts/train.config run ./scripts/train_sfs_adjustable_noiwcdan.nf -resume"

# testing IWCDAN
sbatch -p normal --wrap "/mnt/home/ms4438/programs/nextflow -C ../scripts/test.config run ../scripts/test_sfs_IWCDAN.nf"
sbatch -p normal --wrap "/mnt/home/ms4438/programs/nextflow -C ../scripts/test.config run ../scripts/test_sfs_noIWCDAN.nf"
