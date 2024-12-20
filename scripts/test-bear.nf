params.basedir = "/mnt/home/kc2824/fscratch/popai/bear"
params.sim_dir = "/mnt/scratch/smithfs/cobb/popai/simulations/"

process test_afs {

  memory 4.GB

    input:
    val path

    script:
    """
    /mnt/home/kc2824/popAI/kerry_evo/networks/test.py \
      --json_path $path \
      --source_path $params.sim_dir/bear-secondary-contact-1-1000-test-afs.npz \
      --target_path $params.sim_dir/bear-secondary-contact-ghost-1-1000-test-afs.npz \
      --epoch 50 \
      --outdir test-epoch-50 
    """
}

process test_snp {
    errorStrategy 'ignore'
    memory 16.GB

    input:
    val path

    script:
    """
    /mnt/home/kc2824/popAI/kerry_evo/networks/test.py \
      --json_path $path \
      --source_path $params.sim_dir/bear-secondary-contact-1-1000-test-snp.npz \
      --target_path $params.sim_dir/bear-secondary-contact-ghost-1-1000-test-snp.npz \
      --epoch 50 \
      --outdir test-epoch-50 
    """
}

workflow {
    json_paths = channel.fromPath("${params.basedir}/model1-afs-*/*/config.json")
    test_afs(json_paths)

    // json_paths = channel.fromPath("${params.basedir}/model1-snp-*/*/config.json")
    // test_snp(json_paths)
}