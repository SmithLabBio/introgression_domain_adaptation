### Step 1: Simulate Source Data

## Code
# 1a: Training Data
python src/data/secondaryContact.py secondaryContact1/secondaryContact1.yaml secondaryContact1/secondaryContact1-train 1000
# 1b: Validation Data
python src/data/secondaryContact.py secondaryContact1/secondaryContact1.yaml secondaryContact1/secondaryContact1-val 100
# 1c: Testing Data
python src/data/secondaryContact.py secondaryContact1/secondaryContact1.yaml secondaryContact1/secondaryContact1-test 100

## Differences

# yaml file
* 25 samples to 10 samples
* 20_000 bp to 50_000 bp
* max pop size to 100_000
* div time range from [10_000, 1_000_000] to [50_000, 500_000]

### Step 2: Simulate Ghost Data

## Code
python src/data/ghost.py ghost1/ghost1.yaml ghost1/ghost1-test 100 

## Differences

# yaml file
* 25 samples to 10 samples
* 20_000 bp to 50_000 bp
* max pop size to 100_000
* div time range from [10_000, 1_000_000] to [50_000, 500_000]
* ghost div time range from [100_000, 1_000_000] to [1_000_000, 2_000_000]
* added ghost migration rate range (but kept same as migration rate range between focal populations)

# src/data/ghost.py
* added different range for ghostMigrationRate
* made ghost migration symmetric

### Step 3: Train network without Domain Adaptation.

## Code
python trainConv1d_original.py

## Outputs
* results/original_test_cm.txt (Confusion matrix when applying trained network to test data) (94% accuracy)
* results/original_ghost_cm.txt (Confusion matrix when applying trained network to ghost data) (70% accuracy)
* results/encoded_tSNE_original.png (Encoded space for the source data and the ghost data)

## Changes

# trainConv1d_original.py
* included testing and plotting code in same script as training code.
* save model weights instead of models (because of compatability issues)
* use FineTuning model for training
* use 1500 snps
* don't transpose matrices
* use multichannel arrays

# /src/data/kearsSecondaryContactDataset.py
* added multichannel option and code to convert to multichannel array.

# /src/kerasPlot.py
* code for plotting the encoded space

# /src/models.py
* changes to encoder network in getEncoder (now a 2D convolutional network)
* changes to task network in getTask

### Step 4: Train DANN network

## Code
python trainConv1d_dann.py

## Outputs
* results/dann_test_cm.txt (Confusion matrix when applying trained dann network to test data) (92% accuracy)
* results/dann_ghost_cm.txt (Confusion matrix when applying train dann network to ghost data) (83% accuracy)
* results/encoded_tSNE_dann.png (DANN encoded space for the source data and the ghost data)
* results/training_acc_dann.png (Training and discriminator accuracies across epochs.)
* results/training_loss_dann.png (Training and discriminator losses across epochs.)

## Changes

# trainConv1d_dann.py (relative to trainConv1d_original.py)
* use updateLambda function
* other aspects of training hyperparameters

# /src/kerasPlot.py  (in addition to those described above)
* code for plotting accuracies and losses through time

# /src/models.py (in addition to those described above)
* changes to discriminator network in getDiscriminator

### Step 5: Train CDAN network

## Code
python trainConv1d_cdan.py

## Outputs
* results/cdan_test_cm.txt (Confusion matrix when applying trained CDAN network to test data) (93% accuracy)
* results/cdan_ghost_cm.txt (Confusion matrix when applying train CDAN network to ghost data) (90% accuracy)
* results/encoded_tSNE_cdan.png (CDAN space for the source data and the ghost data)
* results/training_acc_cdan.png (Training and discriminator accuracies across epochs.)
* results/training_loss_cdan.png (Training and discriminator losses across epochs.)

## Changes

# trainConv1d_cdan.py (relative to trainConv1d_dann.py)
* more epochs than used for DANN
* other aspects of training hyperparameters

