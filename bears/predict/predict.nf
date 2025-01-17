
// params.name = "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0"
params.name = "batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0.5"
params.weights_dir = "/mnt/scratch/smithfs/cobb/popai/bear_real5/${params.name}/*/*"

params.outdir = "/mnt/scratch/smithlab/cobb/bears/predictions"

params.filtered_dir = "/mnt/scratch/smithlab/cobb/bears/filtered" 

process predict {
  debug true
  input:
  tuple val(weights), val(parameters), val(populations), val(replicate), val(chromosome) 

  output:
  tuple path("*.csv"), val(parameters)

  script:
  out = "${populations}_${chromosome}_${replicate}"
  """
  /mnt/home/kc2824/domain-adaptation/scripts/predict.py \
  --weights_dir ${weights} \
  --data ${params.filtered_dir}/${populations}_${chromosome}_norm.npy \
  --epoch 50 \
  --outfile ${out}.csv \
  --entryname ${out}
  """
}

workflow {
  replicates = Channel.fromPath(params.weights_dir, type: 'dir')
  chromosomes = channel.fromPath("/mnt/home/kc2824/domain-adaptation/bears/data/selected-scaffolds.txt").splitCsv().map{ it -> it[0] }

  // Split up replicate paths into parts for naming outputs
  input = replicates.map{ it ->
    parts = it.toString().split('/')
    parameters = parts[7]
    population = parts[8]
    replicate = parts[9]
    return [it, parameters, population, replicate]}  

 // Combine with replicate paths with chromosomes 
  .combine(chromosomes)

  // Predict 
  predict(input)

  // Compile all outputs into a single output file
  predict.out.collectFile(name: "${params.name}.csv", sort: { it[0] }, storeDir: params.outdir, newLine: true){ it[0] }
}


