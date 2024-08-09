params.basedir = "/mnt/home/kc2824/fscratch/popai/bear2/"
params.sim_dir = "/mnt/scratch/smithfs/cobb/popai/simulations/"

process test_sfs_norm {

    input:
    tuple val(path)

    script:
    """
    /mnt/home/kc2824/popAI/kerry_evo/networks/test_sfs_bear.py \
      --dir $path \
      --meth finetune \
      --source_path ${params.sim_dir}/bear-secondary-contact-1-1000-test-sfs-norm.npz \
      --target_path ${params.sim_dir}/bear-secondary-contact-ghost-1-1000-test-sfs-norm.npz \
      --epoch 100
    """
}

process test_sfs {

    input:
    tuple val(path)

    script:
    """
    /mnt/home/kc2824/popAI/kerry_evo/networks/test_sfs_bear.py \
      --dir $path \
      --meth finetune \
      --source_path ${params.sim_dir}/bear-secondary-contact-1-1000-test-sfs.npz \
      --target_path ${params.sim_dir}/bear-secondary-contact-ghost-1-1000-test-sfs.npz \
      --epoch 100
    """
}

workflow {
    norm_outputs = channel.fromPath("${params.basedir}/*norm/*", type: "dir")
    outputs = channel.fromPath("${params.basedir}/*sfs/*", type: "dir")
    test_sfs(outputs)
    test_sfs_norm(norm_outputs)
}