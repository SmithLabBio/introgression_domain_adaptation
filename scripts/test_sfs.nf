params.basedir = "/mnt/scratch/smithfs/cobb/popai/general1/"
params.sim_dir = "/mnt/scratch/smithfs/cobb/popai/simulations/"

process test_sfs {

    input:
    tuple val(path)

    script:
    """
    /mnt/home/kc2824/domain-adaptation/scripts/test_sfs.py \
      --dir $path \
      --meth cdan \
      --source_path ${params.sim_dir}/general-secondary-contact-1-1000-test-sfs-norm.npz \
      --target_path ${params.sim_dir}/general-secondary-contact-ghost-1-1000-test-sfs-norm.npz \
      --epoch 50
    """
}

workflow {

    outputs = channel.fromPath("${params.basedir}/*/*", type: "dir")
    test_sfs(outputs)
}