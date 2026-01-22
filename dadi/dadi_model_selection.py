import dadi
import numpy as np
import os
import fire

def secondary_contact(params, ns, pts):
    split, m = params
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.Integration.two_pops(phi, xx, split, 1, 1, m12=0, m21=0)
    phi = dadi.Integration.two_pops(phi, xx, split/2, 1, 1, m12=m, m21=m)
    fs = dadi.Spectrum.from_phi(phi, ns, (xx,xx))
    return fs

def isolation(params, ns, pts):
    split = params[0]
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.Integration.two_pops(phi, xx, split, 1, 1, m12=0, m21=0)
    fs = dadi.Spectrum.from_phi(phi, ns, (xx,xx))
    return fs

def run_dadi(data_path, replicate, model, outdir):
    """
    model: string, either "secondary_contact" or "isolation"
    i: index of the simulation to run in npz file
    """
    data = np.load(data_path)["x"][replicate].squeeze()
    sfs = dadi.Spectrum(data, data_folded=True)

    ns = sfs.sample_sizes

    pts = [max(ns)+20, max(ns)+30, max(ns)+40]

    # Starting value, lower bounds, upper bounds
    split = (250_000, 50_000, 500_000)  # Split time
    m = (2.5, 0.05, 5.0)  # Migration rate

    if model == "isolation":
        params, lower, upper = zip(*[split])
    elif model == "secondary_contact":
        params, lower, upper = zip(*[split, m])

    func_ex = dadi.Numerics.make_extrap_log_func(eval(model))
    p0 = dadi.Misc.perturb_params(params, fold=1, upper_bound=upper, 
                                  lower_bound=lower)

    print("Optimizing...")
    popt = dadi.Inference.optimize_log(p0, sfs, func_ex, pts, 
                                       lower_bound=lower, upper_bound=upper, 
                                       verbose=1, maxiter=10)

    model = func_ex(popt, sfs, ns, pts)
    ll_model = dadi.Inference.ll_multinom(model, sfs)
    

    # best_fit_model = func_ex(popt, ns, pts)
    # popt, ll_model = dadi.Inference.opt()
    # # ll_model = dadi.Inference.ll_multinom(best_fit_model, sfs)
    # print(best_fit_model)
    # print(ll_model)

    # os.makedirs(outdir, exist_ok=True)
    # with open(os.path.join(outdir, f"{model}-{replicate}.txt"), "w") as f:
        # f.write(" ".join(map(str, popt)))
    
data_path = "/mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-1000-test-sfs.npz"
model = "isolation"
# model = "secondary_contact"
rep=0
run_dadi(data_path, rep, model, f"general-secondary-contact-1-1000-test")

# if __name__ == "__main__":
    # fire.Fire(run_dadi)