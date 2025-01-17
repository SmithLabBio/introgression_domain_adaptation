
process variant_call {
  tag {scaffold}
  publishDir params.outdir, mode: "move"

  input:
  val scaffold

  output:
  path "${scaffold}.bcf"

  shell:
  """
  bcftools mpileup \
    -Ou \
    --skip-indels \
    --ignore-RG \
    -d 250 \
    --region ${scaffold} \
    --threads ${task.cpus} \
    --fasta-ref ${params.reference} \
    ${params.bamfiles}/*.bam | 
  bcftools call \
    -mv \
    -Ob \
    -f GQ \
    --threads ${task.cpus} \
    -o ${scaffold}.bcf
  """
}

workflow {
  scaffolds = Channel.fromPath(params.scaffolds).splitCsv().map{it -> it[0]}
  variant_call(scaffolds)

}
