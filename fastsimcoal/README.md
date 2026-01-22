

# Convert from npy to FSC
Covert data from numpy format to FSC

Comment/uncomment file paths at beginning of file.

Then run:
```bash 
python np_to_fsc.py
```



# Sample unlinked snps
For sampling unlinked snps from simulated datasets.

Comment/uncomment file paths at beginning of file.

Then run:
```bash
python sample_unlinked_snps.py
```


# Run fastsimcoal
Runs FastSimCoal

Comment/uncomment file paths in the top of fastsimcoal.nf file.

Then run:
```bash
sbatch --wrap "nextflow -C fastsimcoal.config run fastsimcoal.nf -resume"
```


# FastSimCoal summarization
Produce confusion matrix from fastsimcoal output

Comment/uncomment file paths at beginning of file.

Run: 
```bash
python summarize.py
```