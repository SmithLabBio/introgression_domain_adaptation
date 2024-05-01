## Step 1: simulate data
# training data
python src/data/secondaryContact.py secondaryContact3/secondaryContact3.yaml secondaryContact3/secondaryContact3-train 5000
# validation data
python src/data/secondaryContact.py secondaryContact3/secondaryContact3.yaml secondaryContact3/secondaryContact3-val 100
# test data
python src/data/secondaryContact.py secondaryContact3/secondaryContact3.yaml secondaryContact3/secondaryContact3-test 100
# ghost data
python src/data/ghost.py ghost3/ghost3.yaml ghost3/ghost3-test 100

## Step 2: Training networks
# 1 finetuning
python trainConv2d_original.py
# 2 CDAN


# differences
used shorter sequence length (1,000,000)
used higher mutation rate (1e-8)

megan_working/src/data/secondaryContact.py vs src/data/secondaryContact.py
    - very minor change on how the sampling dict is entered lines 62-64

megan_working/src/data/simulation.py vs src/data/simulation.py
    - no differences

megan_working/src/data/ghost.py vs src/data/ghostSecondaryContact.py
    - no differences

megan_working/src/data/kerasSecondaryContactDataset.py vs exp3/kerasSecondaryContactDataset.py
    - no notable differences

megan_working/models_v2.py vs exp3/conv2d_models.py
    - no notable differences

finetuning network
    - validation data




