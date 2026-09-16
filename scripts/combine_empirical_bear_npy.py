import sys
import glob
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Combine empirical SFS from different population pairs.")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing SFS files.")
    parser.add_argument("-o", "--output", required=True, help="Output file for combined SFS.")

    return parser.parse_args()



def main():
    args = parse_args()
    input_dir = args.input
    output_file = args.output

    # Find all SFS files in the input directory
    sfs_files = glob.glob(f"{input_dir}/*_norm.npz")
    sfs_files = [f for f in sfs_files if "abc" in f]
    
    if not sfs_files:
        print("No SFS files found in the specified directory.")
        sys.exit(1)

    print(f"Found {len(sfs_files)} paths:")
    for p in sfs_files:
        print(f"  {p}")
    
    loaded = {}
    for p in sfs_files:
        with np.load(p, allow_pickle=True) as data:
            loaded[p] = {k: data[k] for k in data.files}

    keys = set(loaded[sfs_files[0]].keys())

    combined = {}
    for key in sorted(keys):
        arrays = [loaded[p][key] for p in sfs_files]

        if arrays[0].ndim == 0:
            item = arrays[0].item()
            if item is None:
                combined[key] = None
                print(f"{key!r}: None in all files -> kept as None")
            else:
                combined[key] = np.array(arrays, dtype=object)
                print(f"{key!r}: scalar field -> stacked shape {combined[key].shape}")
        else:
            combined[key] = np.concatenate(arrays, axis=0)
            print(f"{key!r}: {[a.shape for a in arrays]} -> {combined[key].shape}")

    np.savez(output_file, **combined)

    with np.load(output_file) as combined_output:
        print(combined_output)
        print(combined_output['x'].shape)



if __name__ == "__main__":
    main()