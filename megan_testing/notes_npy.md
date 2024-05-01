
# simulate data
python src/data/simulations_simple.py --yaml secondaryContact3/secondaryContact3.yaml --prefix secondaryContact3/secondaryContact3-train --reps 20000 --seed 1234
python src/data/simulations_simple.py --yaml secondaryContact3/secondaryContact3.yaml --prefix secondaryContact3/secondaryContact3-test --reps 100 --seed 1235
python src/data/simulations_simple.py --yaml secondaryContact3/secondaryContact3.yaml --prefix secondaryContact3/secondaryContact3-val --reps 100 --seed 1237
python src/data/simulations_simple.py --yaml ghost3/ghost3.yaml --prefix ghost3/ghost3-test --reps 100 --seed 1236 --ghost

python src/data/simulations_simple.py --yaml bgs/bgs.yaml --prefix bgs/bgs-test --reps 100 --seed 1236 --bgs --outdir bgs

WHEN CLEANING THIS UP REMOVE THE STATISTICAL TEST AS IT IS NOT CORRECTLY IMPLEMENTED IN PYTHON. WE USED R.