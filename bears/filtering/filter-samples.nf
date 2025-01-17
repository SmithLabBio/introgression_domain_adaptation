params.scaffolds = "/mnt/home/kc2824/domain-adaptation/bears/data/selected-scaffolds.txt"
params.bcf_paths = "/mnt/scratch/smithlab/cobb/bears/variants/"
params.outdir = "/mnt/scratch/smithlab/cobb/bears/filtered/"
params.populations_dir = "/mnt/home/kc2824/domain-adaptation/bears/filtering/"
params.script_dir = "/mnt/home/kc2824/domain-adaptation/bears/filtering/"

populations = [ 
  "brown-abc",
  "brown-alaska",
  "brown-asia",
  "brown-eurasia",
  "brown-eu",
  "brown-hudson",
  "brown-scandanavia",
  "brown-us"]

process filtering {
  // Read bcfs, filter, output vcfs, and output numpy matrix for each chromosome and population pair
  publishDir params.outdir, pattern: "*{.npy,_site_count.txt}", mode: "copy"

  input:
  tuple val(pop0), val(pop1), val(chromosome)

  output:
  tuple val(pop0), val(pop1), val(chromosome), path("*_norm.npy"), emit: files
  path "*_site_count.txt" 

  script:

  outname = "${pop0}_${pop1}_${chromosome}"
  pop0_path = file("${params.populations_dir}/${pop0}.txt")
  pop1_path = file("${params.populations_dir}/${pop1}.txt")

  """
  #!/usr/bin/env fish

  begin
    cat $pop0_path | sed 's|\$| pop0|'
    cat $pop1_path | sed 's|\$| pop1|'
  end > ${outname}.popmap

  set samples (cat $pop0_path $pop1_path | string join ",")

  # Subsample to only include specified samples and include only bi-allelic sites after subsampling
  bcftools view \
    -s \$samples \
    -O u \
    ${params.bcf_paths}/${chromosome}.bcf |
  # Remove sites with any missing genotypes
  bcftools view \
    -O v \
    -e 'GT[*]="mis"' |
  # Remove invariant sites based on the actual genotypes of subsampled 
  bcftools view \
    -O u \
    -o ${outname}.vcf \
    -e 'COUNT(GT="0/0")=N_SAMPLES || COUNT(GT="1/1")=N_SAMPLES || COUNT(GT="2/2")=N_SAMPLES || COUNT(GT="3/3")=N_SAMPLES' 

  set n_sites (bcftools view -H ${outname}.vcf | wc -l)
  echo "Total sites remaining: \$n_sites" > ${outname}_site_count.txt 

  ${params.script_dir}/vcf_to_numpy.py ${outname}.vcf ${outname}.popmap ${outname}_cnts.npy
  ${params.script_dir}/sfs_reformat_to_msprime.py ${outname}_cnts.npy ${outname}_cnts_rfmt.npy
  ${params.script_dir}/normalize_sfs.py ${outname}_cnts_rfmt.npy ${outname}_norm.npy

  """
}

process join_npy {

  publishDir params.outdir, pattern: "*{.npz,_site_count.txt}", mode: "copy"

  input:
  tuple val(pop0), val(pop1), val(chromosomes), path(norm)

  output:
  path "*_norm.npz"

  script:

  outname = "${pop0}_${pop1}"

  """
  #!/usr/bin/env fish

  ${params.script_dir}/join_numpy.py ${outname}_norm.npz ${norm} 

  """
}

workflow {
  samples = Channel.fromList(populations) 
  populations = samples.combine(samples).filter{ it[0] < it[1] } 
  chromosomes = channel.fromPath(params.scaffolds).splitCsv().map{ it -> it[0] }
  comb = populations.combine(chromosomes)
  filtering(comb)
  grouped = filtering.out.files.groupTuple(by:[0,1])
  join_npy(grouped)
}