# Notes for analyses used for NSF proposal

## Step 1: simulate data
python src/data/simulations_simple.py --yaml secondaryContact3/secondaryContact3.yaml --prefix secondaryContact3/secondaryContact3-train --reps 20000 --seed 1234 --outdir secondaryContact3
python src/data/simulations_simple.py --yaml secondaryContact3/secondaryContact3.yaml --prefix secondaryContact3/secondaryContact3-test --reps 100 --seed 1235 --outdir secondaryContact3
python src/data/simulations_simple.py --yaml secondaryContact3/secondaryContact3.yaml --prefix secondaryContact3/secondaryContact3-val --reps 100 --seed 1237 --outdir secondaryContact3
python src/data/simulations_simple.py --yaml ghost3/ghost3.yaml --prefix ghost3/ghost3-test --reps 100 --seed 1236 --ghost --outdir ghost3
python src/data/simulations_simple.py --yaml bgs/bgs.yaml --prefix bgs/bgs-test --reps 100 --seed 1236 --bgs --outdir bgs

## Step 2: Train SFS networks
### original network
python trainConv2d_afs_original.py
### CDAN for ghost
python trainConv2D_afs_cdan_ghost.py
### CDAN for BGS
python trainConv2D_afs_cdan_bgs.py

## Step 3: Train alignment networks
### original network
python trainConv2d_npy_original.py
### CDAN for ghost
python trainConv2d_npy_cdan_ghost.py
### CDAN for BGS
python trainConv2d_npy_cdan_bgs.py

## Step 4: Make figures for Aim 2 and Aim 3
### Aim 2
Rscript plot_aim2.R
### Aim 3
Rscript plot_aim3.R
