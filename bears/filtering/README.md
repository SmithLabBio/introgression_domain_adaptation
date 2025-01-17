# Sample lists
Samples selected from each population to build alignments for:
- brown-abc.txt
- brown-alaska.txt
- brown-asia.txt
- brown-eurasia.txt
- brown-eu.txt
- brown-hudson.txt
- brown-samples.txt
- brown-scandanavia.txt
- brown-us.txt

List of samples included in final analysis:
brown-included-in-final-analyses.txt

# Scripts
## filter-samples.config
Nextflow configuration for filter-samples.nf  

## filter-samples.nf
Nextflow script to filter variants and output alignments

## join_numpy.py
Script to join all chromosome numpy matrices into a single numpy matrix

## normalize_sfs.py
Script to normalize site frequency specture

## site_cnt_summary.py
Number of sites in each alignment

## sfs_corner.py
Move all fixed sites to top corner, consistent with msprime

# popmap.txt
Population assignment of each sample

## Usage
Run filter-samples.nf script which will run other scripts.