
process run_dadi{

    input:
    tuple val(model) val(rep)

    output:
    path "general-secondary-contact-${model}-${rep}.txt" into dadi_params

    script:
    """
    python dadi_model_selection.py \
      "/mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-1000-test-sfs.npz" \
      ${rep} \
      ${model} \
      general-secondary-contact-1-1000-tests
    """
}

workflow {
  replicates = Channel.from(0..999)
  models = Channel.from("isolation", "secondary_contact")
  combined = models.combine(replicates)
  run_dadi(combined)
}