models = ["secondary_contact", "isolation" ]
modeldir = "/mnt/home/kc2824/domain-adaptation/fastsimcoal"

// Data with no ghost introgression, no subsampling of unlinked SNPs
indir = "/mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-100-test-fsc"
outdir = "/mnt/scratch/smithlab/cobb/fastsimcoal/general-secondary-contact-1-100-test-fsc-output"

// Data with no ghost introgression, subsampling of unlinked SNPs 
// indir = "/mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-100-test-fsc-unlinked"
// outdir = "/mnt/scratch/smithlab/cobb/fastsimcoal/general-secondary-contact-1-100-test-fsc-unlinked-output"

// Data with ghost introgression, no subsampling of unlinked SNPs
// indir = "/mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-ghost-1-100-test-fsc"
// outdir = "/mnt/scratch/smithlab/cobb/fastsimcoal/general-secondary-contact-ghost-1-100-test-fsc-output"

// Data with ghost introgression, subsampling of unlinked SNPs
// indir = "/mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-ghost-1-100-test-fsc-unlinked"
// outdir = "/mnt/scratch/smithlab/cobb/fastsimcoal/general-secondary-contact-ghost-1-100-test-fsc-unlinked-output"

process fsc {
  errorStrategy 'retry'
  maxRetries 2
  memory {4.GB * task.attempt}

  publishDir outdir, mode: 'copy'

  input:
    tuple val(model), path(sfs), val(fsc_replicate)
  
  output:
    path "${simulation_replicate}-${model}-${fsc_replicate}"

  script:
    simulation_replicate = sfs.baseName
    """
    hostname 

    # Rename input SFS to match expected format
    cp ${sfs} ${model}_jointMAFpop1_0.obs 
    cp ${modeldir}/${model}.tpl .
    cp ${modeldir}/${model}.est .

    # Run fastsimcoal2
    fsc28 \
      -t ${model}.tpl \
      -e ${model}.est \
      -n 100000 \
      -L 40 \
      -m -M -q -0 \

    mv ${model} ${simulation_replicate}-${model}-${fsc_replicate}
    """
}

workflow {
  fsc_replicates = channel.from(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  sfs = channel.fromPath("${indir}/*.txt")
  models = channel.from(models)
  comb = models.combine(sfs).combine(fsc_replicates)
  fsc(comb)
}