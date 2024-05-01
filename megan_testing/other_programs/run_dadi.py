from src.data.kerasSecondaryContactDataset import Dataset
import dadi
import nlopt
import scipy.stats as stats
import random
import numpy as np
import argparse

def split_secondarycontact(params, ns, pts):
    """
    params = (nu1,nu2,Tmig,m12,m21)
    ns = (n1,n2)

    Split into two populations of specifed size, with migration half way between the split and the present.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    Tmig: Time in the past after migration starts (in units of 2*Na generations) 
    m12: Migration from pop 2 to pop 1 (2*Na*m12)
    m21: Migration from pop 1 to pop 2 (2*Na*m21)
    n1,n2: Sample sizes of resulting Spectrum
    pts: Number of grid points to use in integration.
    """
    nu1,nu2,Tmig,m12, m21 = params

    xx = dadi.Numerics.default_grid(pts)

    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.Integration.two_pops(phi, xx, Tmig//2, nu1, nu2, m12=0, m21=0)
    phi = dadi.Integration.two_pops(phi, xx, Tmig, nu1, nu2, m12=m12, m21=m21)

    fs = dadi.Spectrum.from_phi(phi, ns, (xx,xx))
    return fs

def define_model(ns):

    """Set up the model for inference."""

    # Define the grid points based on the sample size.
    # For smaller data (largest sample size is about <100) [ns+20, ns+30, ns+40] is a good starting point.
    # For larger data (largest sample size is about >=100) or for heavily down projected data [ns+100, ns+110, ns+120] is a good starting point.
    pts_l = [max(ns)+20, max(ns)+30, max(ns)+40]


    model = split_secondarycontact

    # Wrap the demographic model in a function that utilizes grid points which increases dadi's ability to more accurately generate a model frequency spectrum.
    model = dadi.Numerics.make_extrap_func(model)

    return(model, pts_l)

def fit_model(data_fs, model, pts_l, fixed=[None, None, None, None, None]):
    # Define starting parameters
    params = [1, 1, 5, 0.001, 0.001]

    # Define boundaries of optimization.
    # It is a good idea to have boundaries to avoid optimization
    # from trying parameter sets that are time consuming without
    # nessicarily being correct.
    # If optimization infers parameters very close to the boundaries, we should increase them.
    lower_bounds = [1e-2, 1e-2, 0.005, 0, 0]
    upper_bounds = [5, 5, 50, 5, 5]


    # Perturb parameters
    # Optimizers dadi uses are mostly deterministic
    # so we will want to randomize parameters for each optimization.
    # It is recommended to optimize at least 20 time if we only have
    # our local machin, 100 times if we have access to an HPC.
    # If we want a single script to do multiple runs, we will want to
    # start a for loop here
    p0 = dadi.Misc.perturb_params(params, fold=2, upper_bound=upper_bounds,
                                  lower_bound=lower_bounds)

    # Run optimization
    # At the end of the optimization we will get the
    # optimal parameters and log-likelihood.
    # We can modify verbose to watch how the optimizer behaves,
    # what number we pass it how many evaluations are done
    # before the evaluation is printed.

    popt, ll_model = dadi.Inference.opt(p0, data_fs, model, pts_l,
                                        lower_bound=lower_bounds,
                                        upper_bound=upper_bounds,
                                        algorithm=nlopt.LN_BOBYQA,
                                        verbose=100, log_opt = True, maxeval=400, fixed_params=fixed)

    return(popt, ll_model)

def write_results(ll_model, popt, theta0, simple_ll_model, simple_popt, simple_theta0, adj_p_value, p_value, migrationState, output):
    # Write results to fid
    fid = open(output, 'a')
    res = [ll_model] + list(popt) + [theta0] + [simple_ll_model] + list(simple_popt) + [simple_theta0] + [adj_p_value] + [p_value] + [migrationState]
    fid.write('\t'.join([str(ele) for ele in res])+'\n')
    fid.close()

def calc_theta(popt, ns, pts_l, model, data_fs):
    # Calculate the synonymous theta
    model_fs = model(popt, ns, pts_l)
    theta0 = dadi.Inference.optimal_sfs_scaling(model_fs, data_fs)
    return(theta0)


def get_sfs(sfs_array, boot=False):
    # bs sfs
    dsfs = dadi.Spectrum(sfs_array, data_folded=False)
    if boot:
        data_pieces = [(dsfs*(0.5 + (1.5-0.5)/99*ii)).sample() for ii in range(100)]
        all_boot = []
        for boot_ii in range(100):
            # Each bootstrap is made by sampling, with replacement, from our data
            # pieces
            this_pieces = [random.choice(data_pieces) for _ in range(100)]
            all_boot.append(dadi.Spectrum(np.sum(this_pieces, axis=0)))

        return(dsfs, dsfs.sample_sizes, all_boot)

    else:
        return(dsfs, dsfs.sample_sizes, None)

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='Process input and output folders')
    parser.add_argument('--input', dest="input", type=str,
                        help='an input file with data.')
    parser.add_argument('--output', dest="output", type=str,
                        help='an output file.')
    parser.add_argument('--force', action='store_true', help='force overwrite if output file exists (default: False)')
    args = parser.parse_args()

    # read data
    test = Dataset(args.input, 1500, transpose=False, multichannel=True)

    for item in range(test.afs.shape[0]):

        # sfs to dadi
        #dsfs = dadi.Spectrum(test.afs[item,:,:,0])
        #ns = dsfs.sample_sizes
        dsfs, ns, boots = get_sfs(test.afs[item,:,:,0], boot=True)

        # define the model
        model, pts_l = define_model(ns)

        # fit the model
        popt, ll_model = fit_model(dsfs, model, pts_l)

        # calculate theta
        theta0 = calc_theta(popt, ns, pts_l, model, dsfs)

        # fit simple model
        simple_popt, simple_ll_model = fit_model(dsfs, model, pts_l, fixed=[None, None, None, 0, 0])

        # calcualte simple theta
        simple_theta0 = calc_theta(simple_popt, ns, pts_l, model, dsfs)

        # lrt adjust
        adj = dadi.Godambe.LRT_adjust(model, pts_l, boots, popt, dsfs, [3,4], multinom = True, eps=0.01)
        adj_ldrt = adj * 2 * (ll_model - simple_ll_model)
        adj_p_value = 1 - stats.chi2.cdf(adj_ldrt, 2)


        # lrt original
        ldrt = 2 * (ll_model - simple_ll_model)
        p_value = 1 - stats.chi2.cdf(ldrt, 1)
        print(p_value, popt, simple_popt)

        # write results
        write_results(ll_model, popt, theta0, simple_ll_model, simple_popt, simple_theta0, adj_p_value, p_value, test.migrationStates[item], args.output)
