params.basedir = "/mnt/scratch/smithfs/megan/da_revisions/iwcdan_bear/"
params.sim_dir = "/mnt/scratch/smithfs/cobb/popai/simulations/"

process test_sfs {

    input:
    tuple val(path)

    script:
    """
    /mnt/scratch/smithlab/megan/da_revision/scripts/test_sfs_iwcdan_bear.py \
      --dir $path \
      --meth iwcdan \
      --source_path ${params.sim_dir}/bear-secondary-contact-2-1000-test-sfs-norm.npz
    """
}

workflow {

    outputs = channel.fromPath("${params.basedir}/*/*/", type: "dir")
    test_sfs(outputs)
}