sbatch -J sim -p normal --mem 4GB -w "secondary-contact general-secondary-contact-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-1000-train.json 1000 --force"    
sbatch -J sim -p normal --mem 4GB -w "secondary-contact general-secondary-contact-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-100-train.json 100 --force"    
sbatch -J sim -p normal --mem 4GB -w "secondary-contact general-secondary-contact-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-100-val.json 100 --force"    
sbatch -J sim -p normal --mem 4GB -w "secondary-contact general-secondary-contact-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-100-test.json 100 --force"    
sbatch -J sim -p normal --mem 4GB -w "secondary-contact general-secondary-contact-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-1.json 1 --force"    

sbatch -J sim -p normal --mem 4GB -w "ghost-secondary-contact general-secondary-contact-ghost-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-ghost-1-1000-train.json 1000 --force"
sbatch -J sim -p normal --mem 4GB -w "ghost-secondary-contact general-secondary-contact-ghost-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-ghost-1-100-train.json 100 --force"
sbatch -J sim -p normal --mem 4GB -w "ghost-secondary-contact general-secondary-contact-ghost-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-ghost-1-100-val.json 100 --force"
sbatch -J sim -p normal --mem 4GB -w "ghost-secondary-contact general-secondary-contact-ghost-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-ghost-1-100-test.json 100 --force"
sbatch -J sim -p normal --mem 4GB -w "ghost-secondary-contact general-secondary-contact-ghost-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-ghost-1-1.json 1 --force"


sbatch -J sim -p normal --mem 4GB -w "secondary-contact general-secondary-contact-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-20000-train.json 20000 --force"    
sbatch -J sim -p normal --mem 4GB -w "secondary-contact general-secondary-contact-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-1000-test.json 1000 --force"    

sbatch -J sim -p normal --mem 4GB -w "ghost-secondary-contact general-secondary-contact-ghost-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-ghost-1-20000-train.json 20000 --force"
sbatch -J sim -p normal --mem 4GB -w "ghost-secondary-contact general-secondary-contact-ghost-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-ghost-1-1000-test.json 1000 --force"


# Bears
sbatch -J sim -p normal --mem 4GB -w "secondary-contact bear-secondary-contact-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-1-1000-train.json 1000 --force"    
sbatch -J sim -p normal --mem 4GB -w "secondary-contact bear-secondary-contact-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-1-1000-train2.json 1000 --force"    
sbatch -J sim -p normal --mem 4GB -w "secondary-contact bear-secondary-contact-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-1-100-train.json 100 --force"    
sbatch -J sim -p normal --mem 4GB -w "secondary-contact bear-secondary-contact-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-1-100-val.json 100 --force"    
sbatch -J sim -p normal --mem 4GB -w "secondary-contact bear-secondary-contact-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-1-100-test.json 100 --force"    
sbatch -J sim -p normal --mem 4GB -w "secondary-contact bear-secondary-contact-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-1-1.json 1 --force"    

sbatch -J sim -p normal --mem 4GB -w "ghost-secondary-contact bear-secondary-contact-ghost-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-ghost-1-1000-train.json 1000 --force"
sbatch -J sim -p normal --mem 4GB -w "ghost-secondary-contact bear-secondary-contact-ghost-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-ghost-1-100-train.json 100 --force"
sbatch -J sim -p normal --mem 4GB -w "ghost-secondary-contact bear-secondary-contact-ghost-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-ghost-1-36-train.json 36 --force"
sbatch -J sim -p normal --mem 4GB -w "ghost-secondary-contact bear-secondary-contact-ghost-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-ghost-1-100-val.json 100 --force"
sbatch -J sim -p normal --mem 4GB -w "ghost-secondary-contact bear-secondary-contact-ghost-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-ghost-1-100-test.json 100 --force"
sbatch -J sim -p normal --mem 4GB -w "ghost-secondary-contact bear-secondary-contact-ghost-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-ghost-1-1.json 1 --force"


sbatch -J sim -p normal --mem 4GB -w "secondary-contact bear-secondary-contact-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-1-20000-train.json 20000 --force"    
sbatch -J sim -p normal --mem 4GB -w "secondary-contact bear-secondary-contact-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-1-1000-test.json 1000 --force"    

sbatch -J sim -p normal --mem 4GB -w "ghost-secondary-contact bear-secondary-contact-ghost-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-ghost-1-20000-train.json 20000 --force"
sbatch -J sim -p normal --mem 4GB -w "ghost-secondary-contact bear-secondary-contact-ghost-1.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-ghost-1-1000-test.json 1000 --force"

# Bear 2
sbatch -J sim-20000 -m 100 -p normal --mem 4GB -w "secondary-contact bear-secondary-contact-2.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-2-20000-train.json 20000 --force"    
sbatch -J sim-1000 -p normal --mem 4GB -w "secondary-contact bear-secondary-contact-2.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-2-1000-test.json 1000 --force"    
sbatch -J sim-1000 -p normal --mem 4GB -w "secondary-contact bear-secondary-contact-2.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-2-1000-train.json 1000 --force"    
sbatch -J sim-ghost-1000 -p normal --mem 4GB -w "ghost-secondary-contact2 bear-secondary-contact-ghost-2.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-ghost-2-1000-test.json 1000 --force"
sbatch -J sim-ghost-100 -p normal --mem 4GB -w "ghost-secondary-contact2 bear-secondary-contact-ghost-2.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-ghost-2-100-train.json 100 --force"
sbatch -J sim-ghost-36 -p normal --mem 4GB -w "ghost-secondary-contact2 bear-secondary-contact-ghost-2.yaml /mnt/scratch/smithfs/cobb/popai/simulations/bear-secondary-contact-ghost-2-36-train.json 36 --force"