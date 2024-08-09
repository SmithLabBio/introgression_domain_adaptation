params.basedir = "/mnt/home/kc2824/fscratch/popai/bear2"
params.sim_dir = "/mnt/scratch/smithfs/cobb/popai/simulations/"

process test_sfs {

    input:
    tuple val(path), val(meth), val(form)

    script:
    """
    /mnt/home/kc2824/popAI/kerry_evo/networks/test_sfs_bear.py \
      --dir $path \
      --meth $meth \
      --source_path ${params.sim_dir}/bear-secondary-contact-1-1000-test-${form}.npz \
      --target_path ${params.sim_dir}/bear-secondary-contact-ghost-1-1000-test-${form}.npz \
      --outdir test-epoch-75 \
      --epoch 75
    """
}

workflow {

    cdan_sfs = channel.fromPath("${params.basedir}/*-cdan-sfs/*", type: "dir").map{ it -> tuple it, "cdan", "sfs"}
    finetune_sfs = channel.fromPath("${params.basedir}/*cdan-sfs/*", type: "dir").map{ it -> tuple it, "finetune", "sfs"}
    all = cdan_sfs.concat(finetune_sfs)
    test_sfs(all)
}