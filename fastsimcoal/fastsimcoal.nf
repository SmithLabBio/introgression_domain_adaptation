indir = "/mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-100-test-fsc-test"
models = ["secondary_contact", "isolation" ]
modeldir = "/mnt/home/kc2824/domain-adaptation/fastsimcoal"

process fsc {
   memory { 64.GB }

  input:
    tuple val(model), path(sfs)
  script:
    """
    # Rename input SFS to match expected format
    mv ${sfs} ${model}_jointMAFpop1_0.obs 
    cp ${modeldir}/${model}.tpl .
    cp ${modeldir}/${model}.est .

    # Run fastsimcoal2
    fsc28 \
      -t ${model}.tpl \
      -e ${model}.est \
      -n 1 \
      -L 1 \
      -s 0 \
      -m -M -0 -x -q
    """
}

workflow {
  sfs = channel.fromPath("${indir}/*.txt")
  models = channel.from(models)
  comb = models.combine(sfs).view()
  fsc(comb)
}