params.basedir = "/mnt/home/kc2824/fscratch/popai/bear7/"
params.sim_dir = "/mnt/scratch/smithfs/cobb/popai/simulations/"

process test_sfs {

    input:
    tuple val(path)

    script:
    """
    /mnt/home/kc2824/popAI/kerry_evo/networks/test_sfs_bear.py \
      --dir $path \
      --meth cdan \
      --source_path ${params.sim_dir}/bear-secondary-contact-2-1000-test-sfs-norm.npz \
      --target_path ${params.sim_dir}/bear-secondary-contact-ghost-2-1000-test-sfs-norm.npz \
      --epoch 50
    """
}

workflow {
    // outputs = channel.fromPath("${params.basedir}/*/*", type: "dir")
    outputs = channel.fromPath("${params.basedir}/batch16*/*", type: "dir")
    test_sfs(outputs)
}