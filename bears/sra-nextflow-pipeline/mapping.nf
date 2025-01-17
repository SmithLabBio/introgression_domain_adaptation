params.ascp_key_path = "~/.aspera/connect/etc/asperaweb_id_dsa.openssh"
params.ascp_rate_limit = "200M"

process download {
    tag {run}
    maxRetries 3
    errorStrategy { (task.attempt <= maxRetries) ? "retry" : "ignore" }

    input:
        tuple val(sample), val(run), val(read1), val(read2), val(md5_1), val(md5_2)

    output:
        tuple val(sample), val(run), path("*_1.fastq"), path("*_2.fastq")

    script:
    """
    fasterq-dump ${run} --threads ${task.cpus} --split-files --skip-technical 
    """

    // """
    // ascp -l $params.ascp_rate_limit -v -k 3 -T -P 33001 -i $params.ascp_key_path era-fasp@${read1} ${run}_1.fq.gz
    // ascp -l $params.ascp_rate_limit -v -k 3 -T -P 33001 -i $params.ascp_key_path era-fasp@${read2} ${run}_2.fq.gz
    // """

    // """
    // ascp -l $params.ascp_rate_limit -v -k 3 -T -P 33001 -i $params.ascp_key_path era-fasp@${read1} ${run}_1.fq.gz
    // echo "$md5_1 ${run}_1.fq.gz" | md5sum -c -
    // ascp -l $params.ascp_rate_limit -v -k 3 -T -P 33001 -i $params.ascp_key_path era-fasp@${read2} ${run}_2.fq.gz
    // echo "$md5_2 ${run}_2.fq.gz" | md5sum -c -
    // """
    // """
    // wget ${read1} -o ${run}_1.fq.gz --limit-rate ${params.ascp_rate_limit} --continue
    // echo "$md5_1 ${run}_1.fq.gz" | md5sum -c -
    // wget ${read2} -o ${run}_2.fq.gz --limit-rate ${params.ascp_rate_limit} --continue
    // echo "$md5_2 ${run}_2.fq.gz" | md5sum -c -
    // """

    stub:
    """
    touch ${run}_1.fq.gz ${run}_2.fq.gz
    """
}

process fastp {
    tag { run }
    errorStrategy "ignore"

    input:
    tuple val(sample), val(run), path(read1), path(read2)

    output:
    tuple val(sample), val(run), path("${run}_1.trimmed.fq.gz"), path("${run}_2.trimmed.fq.gz"), emit: trimmed
    path("*.fastp_stats.json"), emit: log

    script:
    """
    fastp --thread $task.cpus --in1 ${read1} --in2 ${read2} --out1 ${run}_1.trimmed.fq.gz \
    --out2 ${run}_2.trimmed.fq.gz --json ${run}.fastp_stats.json 
    """

    stub:
    """
    touch ${run}_1.trimmed.fq.gz ${run}_2.trimmed.fq.gz ${run}.fastp_stats.json
    """
}

process fastqc {
    tag { run }
    errorStrategy "ignore"

    input:
    tuple val(sample), val(run), path(read1), path(read2)

    output:
    path "*_fastqc.zip"

    script:
    """
    fastqc --threads $task.cpus --quiet $read1 $read2
    """

    stub:
    """
    touch ${run}_fastqc.zip
    """
}

process multiqc {
    // Needs multiqc v1.18 to parse fastp
    publishDir params.outdir, mode: "symlink"

    input:
    path multiqc_files 

    output:
    path "multiqc_report.html"

    shell:
    """
    multiqc . 
    """

    stub:
    """
    touch multiqc_report.html
    """
}

process bwa_mem {
    tag { run }
    maxRetries 3 
    errorStrategy { (task.attempt <= maxRetries) ? "retry" : "ignore" }
    publishDir params.outdir, mode: "move", overwrite:true

    input:
    tuple val(sample), val(run), path(read1), path(read2)

    output:
    tuple val(sample), val(run), path("${run}.bam")

    script:
    """
    bwa-mem2 mem -p -t $task.cpus $params.index $read1 $read2 | samtools sort --threads $task.cpus -o ${run}.bam
    """

    stub:
    """
    touch ${run}.bam
    """
}

process samtools_merge_index {
    tag {sample}
    errorStrategy "ignore"
    publishDir params.outdir, mode: "move", overwrite:true

    input: 
    tuple val(sample), path(bam_files)

    output:
    tuple val(sample), path("${sample}.bam"), path("${sample}.bam.bai")

    script:
    // if (bam_files.size() > 1)
    // """
    // samtools merge --threads $task.cpus -o ${sample}.bam $bam_files
    // samtools index --threads $task.cpus ${sample}.bam 
    // """
    // else
    """
    mv $bam_files ${sample}.bam
    samtools index --threads $task.cpus ${sample}.bam 
    """

    stub:
    if ( bam_files.size() > 1 )
    """
    cat $bam_files > ${sample}.bam
    touch ${sample}.bam.bai
    """
    else
    """
    mv $bam_files ${sample}.bam
    touch ${sample}.bam.bai
    """
}



workflow  {
    runs = Channel.fromPath(params.samplesheet).splitCsv(header:true)
    download(runs)
    fastp(download.out)
    fastqc(fastp.out.trimmed)
    bwa_mem(fastp.out.trimmed)

    // Merge bam files by sample and index bam file
    grouped = bwa_mem.out.map{ it -> tuple(it[0], it[2]) }.groupTuple()
    // samtools_merge_index(grouped)

    // Collect all outputs and run QC
    log_files = Channel.empty().mix(fastqc.out).mix(fastp.out.log).collect()
    multiqc(log_files)
}