
set outdir "/mnt/home/kc2824/fscratch/popai/output"

./summarize.py \
  $outdir"/afs-means.txt" \
  $outdir"/afs-orig" \
  $outdir"/afs-cdan/model1-rate_ratio_0.1-rate_1e-4" \
  $outdir"/afs-cdan/model1-rate_ratio_1-rate_1e-4" \
  $outdir"/afs-cdan/model1-rate_ratio_5-rate_1e-4" \
  $outdir"/afs-cdan/model1-rate_ratio_10-rate_1e-4" 

./summarize.py \
  $outdir"/snp-means.txt" \
  $outdir"/snp-orig" \
  $outdir"/snp-cdan/model1-rate_ratio_0.1-rate_1e-4" \
  $outdir"/snp-cdan/model1-rate_ratio_1-rate_1e-4" \
  $outdir"/snp-cdan/model1-rate_ratio_5-rate_1e-4" \
  $outdir"/snp-cdan/model1-rate_ratio_10-rate_1e-4" \
  $outdir"/snp-cdan/model1-rate_ratio_0.1-rate_1e-5" \
  $outdir"/snp-cdan/model1-rate_ratio_1-rate_1e-5" \
  $outdir"/snp-cdan/model1-rate_ratio_5-rate_1e-5" \
  $outdir"/snp-cdan/model1-rate_ratio_10-rate_1e-5" 



./summarize.py \
  $outdir"/afs-max.txt" \
  $outdir"/afs-orig" \
  $outdir"/afs-cdan/model1-rate_ratio_0.1-rate_1e-4" \
  $outdir"/afs-cdan/model1-rate_ratio_1-rate_1e-4" \
  $outdir"/afs-cdan/model1-rate_ratio_5-rate_1e-4" \
  $outdir"/afs-cdan/model1-rate_ratio_10-rate_1e-4" \
  --stat max

./summarize.py \
  $outdir"/snp-max.txt" \
  $outdir"/snp-orig" \
  $outdir"/snp-cdan/model1-rate_ratio_0.1-rate_1e-4" \
  $outdir"/snp-cdan/model1-rate_ratio_1-rate_1e-4" \
  $outdir"/snp-cdan/model1-rate_ratio_5-rate_1e-4" \
  $outdir"/snp-cdan/model1-rate_ratio_10-rate_1e-4" \
  $outdir"/snp-cdan/model1-rate_ratio_0.1-rate_1e-5" \
  $outdir"/snp-cdan/model1-rate_ratio_1-rate_1e-5" \
  $outdir"/snp-cdan/model1-rate_ratio_5-rate_1e-5" \
  $outdir"/snp-cdan/model1-rate_ratio_10-rate_1e-5" \
  --stat max
