import yaml
import torch
import msprime
from torch.distributions.uniform import Uniform
import numpy as np
from tskit import TreeSequence, Variant
from scipy.spatial.distance import euclidean
import csv
import os
import argparse
import sys
import tskit
import pyslim
import random

def getGenotypeMatrix(ts: TreeSequence, nSnps: int, transpose=False, multichannel=False) -> np.ndarray:
    var = Variant(ts, samples=ts.samples()) 
    if transpose:
        shape = (nSnps, len(ts.samples()))
    else:
        shape = (len(ts.samples()), nSnps)
    mat = np.empty(shape=shape)
    for site in range(nSnps):
        try:
            var.decode(site)
            if transpose:
                mat[site, :] = var.genotypes
            else:
                mat[:, site] = var.genotypes
        except:
            list_of_neg_ones = [-1] * len(ts.samples())
            if transpose:
                mat[site, :] = list_of_neg_ones
            else:
                mat[:, site] = list_of_neg_ones
            
    if multichannel:
        channel1 = mat[:len(ts.samples())//2, :]
        channel2 = mat[len(ts.samples())//2:, :]
        ref_row = channel1[0]
        distances=[euclidean(ref_row, row) for row in channel1]
        sorted_indices = np.argsort(distances)
        channel1_sorted = channel1[sorted_indices]
        ref_row = channel2[0]
        distances=[euclidean(ref_row, row) for row in channel2]
        sorted_indices = np.argsort(distances)
        channel2_sorted = channel2[sorted_indices]
        mat = np.stack([channel1_sorted, channel2_sorted], axis=-1) # might want a different axis?

    return mat

def parse_yaml(file_path, ghost):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    n_samples = config.get('nSamples')
    sequence_length = config.get('sequenceLength')
    recombination_rate = float(config.get('recombinationRate'))
    mutation_rate = float(config.get('mutationRate'))
    migration_rate_range = [float(x) for x in config.get('migrationRateRange')]
    pop_size_range = config.get('popSizeRange')
    div_time_range = config.get('divTimeRange')
    ghost_div_time_range = config.get('ghostDivTimeRange')
    ghost_mig_time_range = config.get('ghostMigTimeRange')
    ghost_proportion_range = config.get('ghostProportionRange')
    nSNPs = config.get('nSNPs')

    # Validation
    if not isinstance(n_samples, int):
        raise ValueError("n_samples should be an integer.")
    if not isinstance(sequence_length, int):
        raise ValueError("sequence_length should be an integer.")
    if not isinstance(recombination_rate, float):
        raise ValueError("recombination_rate should be a float.")
    if not isinstance(mutation_rate, float):
        raise ValueError("mutation_rate should be a float.")
    if not all(isinstance(rate, float) for rate in migration_rate_range):
        raise ValueError("migration_rate_range should be a list of floating point values.")
    if not all(isinstance(size, int) for size in pop_size_range):
        raise ValueError("pop_size_range should be a list of integers.")
    if not all(isinstance(time, int) for time in div_time_range):
        raise ValueError("div_time_range should be a list of integers.")
    if not isinstance(nSNPs, int):
        raise ValueError("nSNPs should be an integer.")

    if ghost:
        if not all(isinstance(time, int) for time in ghost_div_time_range):
            raise ValueError("ghost_div_time_range should be a list of integers.")
        if not all(isinstance(time, int) for time in ghost_mig_time_range):
            raise ValueError("ghost_mig_time_range should be a list of integers.")
        if not all(isinstance(rate, float) for rate in ghost_proportion_range):
            raise ValueError("ghost_proportion_range should be a list of floating point values.")

        return {
            'n_samples': n_samples,
            'sequence_length': sequence_length,
            'recombination_rate': recombination_rate,
            'mutation_rate': mutation_rate,
            'migration_rate_range': migration_rate_range,
            'pop_size_range': pop_size_range,
            'div_time_range': div_time_range,
            'ghost_div_time_range': ghost_div_time_range,
            'ghost_mig_time_range': ghost_mig_time_range,
            'ghost_proportion_range': ghost_proportion_range,
            'nSNPs': nSNPs
        }
    else:
        return {
            'n_samples': n_samples,
            'sequence_length': sequence_length,
            'recombination_rate': recombination_rate,
            'mutation_rate': mutation_rate,
            'migration_rate_range': migration_rate_range,
            'pop_size_range': pop_size_range,
            'div_time_range': div_time_range,
            'nSNPs': nSNPs
        }


def get_demography(parsed_config, seeds, ix, reps, ghost, outdir):
    
    torch.manual_seed(seeds[ix])

    # draw always used parameters from priors
    popSize = int(torch.randint(parsed_config['pop_size_range'][0], parsed_config['pop_size_range'][1], (1,)).item())
    divTime = int(torch.randint(parsed_config['div_time_range'][0], parsed_config['div_time_range'][1], (1,)).item())

    # add baseline demography
    demography = msprime.Demography()
    demography.add_population(name="d", initial_size=popSize) # sampled pop 1
    demography.add_population(name="e", initial_size=popSize) # sampled pop 2
    demography.add_population(name="c", initial_size=popSize) # ancestral pop of 1 and 2
    demography.add_population_split(time=divTime, derived=["d", "e"], ancestral="c")

    # add migration as needed
    half = reps // 2
    if ix >= half:
        migrationRate = Uniform(parsed_config['migration_rate_range'][0], parsed_config['migration_rate_range'][1]).sample().item()
        demography.add_symmetric_migration_rate_change(populations=["d", "e"], 
                time=0, rate=migrationRate)
        demography.add_symmetric_migration_rate_change(populations=["d", "e"], 
                time=divTime//2, rate=0)
        migrationState = 1
    
    else: 
        migrationRate = 0
        migrationState = 0
    
    # add ghost as needed
    if ghost:
        ghostDivTime = int(torch.randint(parsed_config['ghost_div_time_range'][0], parsed_config['ghost_div_time_range'][1], (1,)).item())
        ghostMigTime = int(torch.randint(parsed_config['ghost_mig_time_range'][0], parsed_config['ghost_mig_time_range'][1], (1,)).item())
        ghostProportion = Uniform(parsed_config['ghost_proportion_range'][0], parsed_config['ghost_proportion_range'][1]).sample().item()
        demography.add_population(name="b", initial_size=popSize) # add ghost
        demography.add_population(name="a", initial_size=popSize) # add ancestor
        demography.add_population_split(time=ghostDivTime, derived=["b", "c"], ancestral="a")
        demography.add_mass_migration(time=ghostMigTime, source="d", dest="b", proportion=ghostProportion)

    # sort events and simulate data
    demography.sort_events()
    ts = msprime.sim_ancestry(samples={"d": parsed_config["n_samples"], "e": parsed_config["n_samples"]},
                              demography=demography,
                              random_seed=torch.randint(0, 2**32, (1,)).item(),
                              sequence_length=parsed_config["sequence_length"],
                              recombination_rate=parsed_config['recombination_rate'])
    mts = msprime.sim_mutations(ts, rate=parsed_config['mutation_rate'], 
            random_seed=torch.randint(0, 2**32, (1,)).item())
    
    # create afs
    afs = mts.allele_frequency_spectrum(sample_sets=[mts.samples(0), mts.samples(1)], span_normalise=False, polarised=True)

    # create numpy matrix
    matrix = getGenotypeMatrix(mts, parsed_config['nSNPs'], transpose=False, multichannel=True)
                               
    return([popSize, divTime, migrationRate], matrix, afs, migrationState)

def generate_genomic_regions(sequence_length):
    regions = []
    total_length = 0

    while total_length < sequence_length:
        length = random.randint(1000, 3000)
        length = min(length, sequence_length - total_length)
        sel_type = random.sample(['g1','g2','g3'], k=1)[0]
        regions.append((total_length, total_length + length, sel_type))
        total_length += length

    return regions

def generate_recombination_map(sequence_length, scale):
    regions = []
    total_length = 0

    while total_length < sequence_length:
        length = random.randint(1000, 3000)
        length = min(length, sequence_length - total_length)
        sel_type = random.sample([0*scale,1e-9*scale,1e-8*scale], k=1)[0]
        regions.append((total_length + length-1, sel_type))
        total_length += length

    return regions


def write_tuples_to_file(tuples_list, filename):
    with open(filename, 'w') as file:
        for tpl in tuples_list:
            file.write(f"{tpl[0]}\t{tpl[1]}\t{tpl[2]}\n")

def write_rec_tuples_to_file(tuples_list, filename):
    with open(filename, 'w') as file:
        for tpl in tuples_list:
            file.write(f"{tpl[0]}\t{tpl[1]}\n")


def get_demography_slim(parsed_config, seeds, ix, reps, ghost, slim, outdir):
    
    torch.manual_seed(seeds[ix])

    # draw always used parameters from priors
    popSize = int(torch.randint(parsed_config['pop_size_range'][0], parsed_config['pop_size_range'][1], (1,)).item())
    divTime = int(torch.randint(parsed_config['div_time_range'][0], parsed_config['div_time_range'][1], (1,)).item())
    propdel = Uniform(0.1, 0.8).sample().item()
    scale = 100


    # add migration as needed
    half = reps // 2
    if ix >= half:
        migrationRate = Uniform(parsed_config['migration_rate_range'][0], parsed_config['migration_rate_range'][1]).sample().item()
        migrationState = 1
    
    else: 
        migrationRate = 0
        migrationState = 0

    # generate genomic map
    genomic_regions = generate_genomic_regions(parsed_config['sequence_length'])
    write_tuples_to_file(genomic_regions, '%s/genomic_map_%r.txt' % (outdir, ix))

    # generate recombination map
    recombination_regions = generate_recombination_map(parsed_config['sequence_length'], scale)
    write_rec_tuples_to_file(recombination_regions, '%s/recombination_map_%r.txt' % (outdir, ix))

    # simulate data
    if migrationState == 0:
        command = f"{slim} -d rep={ix} -d mu={parsed_config['mutation_rate']*scale} -d seqlen={parsed_config['sequence_length']} -d ne={popSize//scale} -d tdiv={divTime//scale} -d propdel={propdel} -d recrate={parsed_config['recombination_rate']*scale} ~/Documents/GitHub/popAI/megan_testing/slim/nomig_bgs.slim > nomig_{ix}.sliminfo.txt"
        prefix = 'nomig_bgs'

    elif migrationState == 1:
        command = f"{slim} -d rep={ix} -d mu={parsed_config['mutation_rate']*scale} -d seqlen={parsed_config['sequence_length']} -d ne={popSize//scale} -d tdiv={divTime//scale} -d propdel={propdel} -d recrate={parsed_config['recombination_rate']*scale} -d migrate={migrationRate*scale} ~/Documents/GitHub/popAI/megan_testing/slim/mig_bgs.slim > mig_{ix}.sliminfo.txt"
        prefix = 'mig_bgs'

    startdir = os.getcwd()
    os.chdir(outdir)
    os.system(command)
    os.chdir(startdir)

    # load tree sequence
    ts = tskit.load("%s/%s_%s.trees" % (outdir, prefix, ix))
    ts = pyslim.recapitate(ts, recombination_rate=parsed_config['recombination_rate']*scale, ancestral_Ne=popSize//scale)

    # sample 10 individuals from each population
    alive = pyslim.individuals_alive_at(ts, 0)
    pops = [np.where(
       ts.individual_populations[alive] == k)[0] for k in [1,2]]
    sample_inds = [np.random.choice(pop, parsed_config['n_samples'], replace=False) for pop in pops]
    keep_inds = np.concatenate(sample_inds)

    # simplify tree sequence to contain only ten individuals per population
    keep_nodes = []
    for k in keep_inds:
        keep_nodes.extend(ts.individual(k).nodes)

    ts = ts.simplify(keep_nodes)

    ## add neutral mutations
    #mutrate_neutral = parsed_config['mutation_rate']*scale * (1-propdel)
    #mts = msprime.mutate(ts, rate=mutrate_neutral, keep=True)
    mts = ts

    # create afs
    afs = mts.allele_frequency_spectrum(sample_sets=[mts.samples(0), mts.samples(1)], span_normalise=False, polarised=True)

    # create numpy matrix
    matrix = getGenotypeMatrix(mts, parsed_config['nSNPs'], transpose=False, multichannel=True)
                               
    return([popSize, divTime, migrationRate], matrix, afs, migrationState)

def write_params_to_file(params, outfile):
    # Define the header
    header = ['popSize', 'divtime', 'migRate']
    
    # Writing data to the file
    with open(outfile, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')

        # Write the header
        writer.writerow(header)

        # Write the data
        for row in params:
            writer.writerow(row)

def write_matrices_to_file(matrices, outfile):

    combined_matrix = np.stack(matrices)
    np.save(outfile, combined_matrix, allow_pickle=True)

def write_afs_to_file(afs, outfile):

    combined_matrix = np.stack(afs)
    np.save(outfile, combined_matrix, allow_pickle=True)

def write_labels_to_file(labels, outfile):

    combined_matrix = np.stack(labels)
    np.save(outfile, combined_matrix, allow_pickle=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simulate data and save to files.')
    parser.add_argument('--yaml', type=str, default='secondaryContact3/secondaryContact3.yaml',
                        help='Path to the YAML configuration file')
    parser.add_argument('--prefix', type=str, default='secondaryContact3/secondaryContact3-train',
                        help='Output file prefix')
    parser.add_argument('--reps', type=int, default=None,
                        help='Number of repetitions')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--ghost', action='store_true', help='Include ghost option (default: False)')
    parser.add_argument('--bgs', action='store_true', help='Include BGS option (default: False)')
    parser.add_argument('--slim', type=str, help='Path to SLiM', default="slim")
    parser.add_argument('--outdir', type=str, help='Path to store slim results')

    args = parser.parse_args()

    # parse the yaml file
    try:
        parsed_config = parse_yaml(args.yaml, args.ghost)
    except ValueError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

    # get random seeds
    if args.seed: 
        torch.manual_seed(args.seed)
    else:
        args.seed = torch.initial_seed() 
    randomSeeds = torch.randint(0, 2**32, (args.reps,))

    # run simulaations
    all_params = []
    all_matrices = []
    all_afs = []
    all_labels = []
    for i in range(args.reps):
        if args.slim:
            parameters, matrix, afs, label = get_demography_slim(parsed_config, randomSeeds, i, reps=args.reps, ghost=args.ghost, slim=args.slim, outdir = args.outdir)
        else:
            parameters, matrix, afs, label = get_demography(parsed_config, randomSeeds, i, reps=args.reps, ghost=args.ghost)
        
        all_params.append(parameters)
        all_matrices.append(matrix)
        all_afs.append(afs)
        all_labels.append(label)
    
    # save parameters
    write_params_to_file(params=all_params, outfile=f"{args.prefix}_params.tsv")

    # save numpy arrays
    write_matrices_to_file(matrices = all_matrices, outfile=f"{args.prefix}_matrices.npy")
        
    # save SFS
    write_afs_to_file(afs = all_afs, outfile = f"{args.prefix}_afs.npy")

    # save labels
    write_labels_to_file(labels=all_labels, outfile = f"{args.prefix}_labels.npy")
