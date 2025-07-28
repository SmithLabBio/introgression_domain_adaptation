indir = "/mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-100-test-fsc"
models = ["secondary_contact", "isolation" ]
modeldir = "/mnt/home/kc2824/domain-adaptation/fastsimcoal"
outdir = "/mnt/scratch/smithlab/cobb/fastsimcoal/general-secondary-contact-1-100-test-fsc-output"

process fsc {
  memory { 4.GB }

  publishDir outdir, mode: 'copy'

  input:
    tuple val(model), path(sfs), val(fsc_replicate)
  
  output:
    path "${simulation_replicate}-${model}-${fsc_replicate}"

  script:
    simulation_replicate = sfs.baseName
    """
    # Rename input SFS to match expected format
    mv ${sfs} ${model}_jointMAFpop1_0.obs 
    cp ${modeldir}/${model}.tpl .
    cp ${modeldir}/${model}.est .

    # Run fastsimcoal2
    fsc28 \
      -t ${model}.tpl \
      -e ${model}.est \
      -n 100000 \
      -L 40 \
      -m -M
    
    mv ${model} ${simulation_replicate}-${model}-${fsc_replicate}
    """
}

workflow {
  fsc_replicates = channel.from(1, 2)
  sfs = channel.fromPath("${indir}/*.txt")
  models = channel.from(models)
  comb = models.combine(sfs).combine(fsc_replicates)
  fsc(comb)
}