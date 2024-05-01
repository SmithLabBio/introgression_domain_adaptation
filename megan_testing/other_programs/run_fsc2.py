from src.data.kerasSecondaryContactDataset import Dataset
import os

# path to fsc
fsc2 = '~/Documents/Programs/fsc26_mac64/fsc26'
input_data = "secondaryContact3/secondaryContact3-mini.json"
run_fsc2 = "fsc2_results"
# read data
test = Dataset(input_data, 1500, transpose=False, multichannel=True)

for item in range(test.afs.shape[0]):

    # write sfs to file for fsc2
    os.chdir(run_fsc2)
    with open('Migration_jointMAFpop1_0.obs', 'w') as f:
        f.write('1 observation\n')
        column_names = [f"d0_{x}\t" for x in range(20)]
        f.write('\t')
        f.write(''.join(column_names))
        f.write('\n')
        for i in range(20):
            f.write(f"d0_{i}\t")
            to_write = '\t'.join([str(element) for element in test.afs[1,i,:,0]])
            f.write(to_write)
            f.write('\n')
    os.system(f"{fsc2} -t ../Migration.tpl -e ../Migration.est --msfs --removeZeroSFS -M -n 100000")
    os.chdir('../')
