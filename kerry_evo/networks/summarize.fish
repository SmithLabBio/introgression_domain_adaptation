set dir "/mnt/home/kc2824/fscratch/popai/output2" 
./summarize.py "$dir/model1-afs*" test-epoch-50 $dir/summary-afs.txt
./summarize.py "$dir/model1-afs*" test-epoch-50 $dir/summary-afs-max.txt --stat max
./summarize.py "$dir/model1-snp*" test-epoch-50 $dir/summary-snp.txt
./summarize.py "$dir/model1-snp*" test-epoch-50 $dir/summary-snp-max.txt --stat max

set dir "/mnt/home/kc2824/fscratch/popai/bear" 
./summarize.py "$dir/model1-afs*" test-epoch-50 $dir/summary-afs.txt
./summarize.py "$dir/model1-afs*" test-epoch-50 $dir/summary-afs-max.txt --stat max
# ./summarize.py "$dir/model1-snp*" test-epoch-50 $dir/summary-snp.txt
# ./summarize.py "$dir/model1-snp*" test-epoch-50 $dir/summary-snp-max.txt --stat max