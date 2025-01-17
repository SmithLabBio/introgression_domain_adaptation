set basedir "/mnt/scratch/smithlab/cobb/bears"
set workdir "$basedir/nf-call"
set outdir "$basedir/variants"
set reference "$basedir/brown-bear-genome/GCF_023065955.2_UrsArc2.0_genomic.fna"
set bamfiles "$basedir/mapped/"
set scaffolds "/mnt/home/kc2824/bears/data/selected-scaffolds.txt" 

sub -n calling -w "nextflow -C calling.config run -w $workdir calling.nf --outdir $outdir --reference $reference --scaffolds $scaffolds --bamfiles $bamfiles"