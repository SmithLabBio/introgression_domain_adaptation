
params.indir = "/mnt/scratch/smithlab/cobb/bears/filtered"
params.pcadir = "/mnt/home/kc2824/bears/pca"
params.popmap = "/mnt/home/kc2824/bears/filtering/popmap.txt"
params.outdir = "/mnt/scratch/smithlab/cobb/bears/pca"


process pca {
  input:
  val chrom 

  output:
  val chrom

  script:
  """
  VCF2PCACluster \
    -InVCF ${params.indir}/${chrom}.vcf \
    -OutPut ${params.outdir}/${chrom} \
    -InSampleGroup ${params.popmap}
  """

}

process plot {
  input:
  val chrom 

  script:
  """
  ${params.pcadir}/plot_pca.py \
    --input ${params.outdir}/${chrom}.eigenvec \
    --output ${params.outdir}/${chrom}
  """
}

workflow  {
  chromosomes = Channel.fromPath("/mnt/home/kc2824/bears/data/selected-scaffolds.txt").splitCsv().map{ it -> it[0] } 
  pca(chromosomes)
  plot(pca.out)

}
