params.basedir = "/mnt/scratch/smithfs/megan/da_revisions/pseudo_iwcdan_v2/"
params.sim_dir = "/mnt/scratch/smithfs/cobb/popai/simulations/"

process test_sfs {

    input:
    tuple val(path)

    script:
    """
    /mnt/scratch/smithlab/megan/da_revision/scripts/test_sfs_iwcdan.py \
      --dir $path \
      --meth iwcdan \
      --source_path ${params.sim_dir}/general-secondary-contact-1-1000-test-sfs-norm.npz
    """
}

workflow {

    outputs = channel.fromPath("${params.basedir}/*/*/", type: "dir")
    test_sfs(outputs)
}