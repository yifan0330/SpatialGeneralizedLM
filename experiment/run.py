from data_simulation import simulated_data, GRF_simulated_data, SpatialHomo_simulated_data, SubjectHomo_simulated_data, Biobank_data
from bspline import B_spline_bases, RandomFourierFeatures_3D, QMCFeatures_3D
from regression import BrainRegression_full, BrainRegression_Approximate
from inference import BrainInference_full, BrainInference_Approximate, BrainInference_UKB
from util import create_lesion_mask, preprocess_Z
from plot import plot_brain, save_nifti
from nilearn.datasets import load_mni152_template
import nibabel as nib
from absl import logging 
import numpy as np
import scipy
import argparse
import torch 
import dask
import time
import sys
import os


# Example usage:

def parse_int_or_str(value):
    try:
        return int(value)
    except ValueError:
        return value
        
def get_args():
    parser = argparse.ArgumentParser(description="Arguments for data generation, regression, and inference")
    # Boolean flags
    parser.add_argument('--simulated_dset', type=lambda x: x.lower() == 'true', default=True,
                            help="Use simulationed dataset (True or False, default: True)")
    parser.add_argument('--homogeneous', type=lambda x: x.lower() == 'true', default=True,
                            help="Set homogeneous underlying function (True or False, default: True)")
    parser.add_argument("--UKB_subject", type=int, default=13677,
                        help="number of subjects to use from UKB dataset")
    # modelling stages
    parser.add_argument('--run_data_generation', type=lambda x: x.lower() == 'true', default=True,
                        help="Run data generation (default: True)")
    parser.add_argument('--run_regression', type=lambda x: x.lower() == 'true', default=True,
                        help="Run regression (default: True)")
    parser.add_argument('--run_inference', type=lambda x: x.lower() == 'true', default=True,
                        help="Run inference (default: True)")
    # Model parameters
    parser.add_argument('--full_model', type=lambda x: x.lower() == 'true', default=True,
                        help="Use full model or memory-efficient approximate model for regression (default: True)")
    parser.add_argument('--gradient_mode', type=str, default="approximate", help="Gradient mode for optimisation (default: offload)")
    parser.add_argument('--preconditioner_mode', type=str, default="approximate", help="Preconditioner mode for optimisation (default: approximate)")
    parser.add_argument('--model', type=str, default="SpatialBrainLesion",
                        help="Type of stochastic model (default: Poisson)")
    parser.add_argument('--regression_terms', nargs='+', type=str, help="Regression terms (default: ['multiplicative', 'additive'])")
    parser.add_argument('--link_func', type=str, default="logit", help="Link function for intensity function (default: logit)")
    parser.add_argument('--polynomial_order', type=int, default=1, help="Polynomial order for spatial basis (default: 3)") 
    parser.add_argument('--marginal_dist', type=str, default="Bernoulli", help="Marginal distribution at each spatial location (default: Bernoulli)")
    parser.add_argument('--std_params', type=float, default=0.1, help="Standard deviation of Gaussian parameters (default: 0.1)")
    parser.add_argument('--lr', type=float, default=1, help="Learning rate for optimisation (default: 0.1)")
    parser.add_argument('--tol', type=float, default=1e-7, help="Tolerance for optimisation (default: 1e-7)")
    parser.add_argument('--iter', type=int, default=1e4, help="Number of iterations for optimisation (default: 100)")
    parser.add_argument('--firth_penalty', type=lambda x: x.lower() == 'true', default=False, help="Use Firth penalty for regression (default: False)")

    # Inference parameters
    parser.add_argument('--contrast_vector', nargs='+', type=int, default=None, help="Contrast vector for t-test (default: None)")
    parser.add_argument('--contrast_name', type=str, default=None, help="Contrast names for t-test (default: None)")
    parser.add_argument('--inference_method', type=str, default="FI", help="Inferential method (default: FI)")

    # General options
    parser.add_argument('--gpus', type=str, default="0", help="GPU device (default: 0)")
    parser.add_argument('--space_dim', type=parse_int_or_str, default=1, 
                        help="Dimension of simulation space (default: 1)")
    parser.add_argument('--n_group', type=int, default=1,
                        help="Number of groups (default: 1)")
    parser.add_argument('--group_names', nargs='+', type=str, default=None,
                        help="Name of groups (default: Group_1, Group_2, etc.)")
    parser.add_argument('--n_subject', nargs='+',type=int, default=[1000],
                        help="Number of subjects (default: [1000])")
    parser.add_argument('--spacing', type=int, default=10,
                        help="Spacing for B-spline basis (default: 10)")
    parser.add_argument('--lesion_per_subject', nargs='+', type=int, default=[10],
                        help="Number of lesions per subject (default: 10). Accepts a single integer or a comma-separated list of integers.")
    parser.add_argument('--random_seed', type=int, default=0,
                        help="Random seed for GRF simulation (default: 0)")

    # Auxiliary variables
    parser.add_argument('--n_auxiliary', type=int, default=2,
                        help="Number of auxiliary variables (default: 2)")
    parser.add_argument('--std_auxiliary', type=float, default=1.0, 
                        help="Standard deviation of auxiliary variables (default: 1.0)")
    parser.add_argument('--n_samples', type=int, default=100, help="Number of samples for Monte Carlo approximation (default: 100)")
    return parser.parse_args()

args = get_args()

# Validate model type
SUPPORTED_MODELS = {"MassUnivariateRegression", "SpatialBrainLesion"}
if args.model not in SUPPORTED_MODELS:
    raise ValueError(f"Model '{args.model}' is not supported. Supported models are: {', '.join(sorted(SUPPORTED_MODELS))}")
if args.group_names and len(args.group_names) != args.n_group:
    raise ValueError(f"Number of group names ({len(args.group_names)}) does not match number of groups ({args.n_group})")



simulated_dset = args.simulated_dset
n_subjects_whole_UKB = 13677
UKB_subject = args.UKB_subject if not args.simulated_dset else None
homogeneous = args.homogeneous
space_dim = args.space_dim
spacing = args.spacing
n_group = args.n_group
group_names = args.group_names if args.group_names else [f"Group_{i+1}" for i in range(n_group)]
n_subject = args.n_subject
lesion_per_subject = args.lesion_per_subject
polynomial_order = args.polynomial_order
model = args.model
marginal_dist = args.marginal_dist
lr = args.lr
tolerance_change = args.tol
iter = args.iter

# raise an error if n_group does not match length of n_subject
if len(n_subject) != n_group:
    raise ValueError(f"Number of subjects: {len(n_subject)} does not match number of groups: {n_group}")


# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
    device = 'cuda'
else:
    device = 'cpu'

data_dir = "/well/nichols/projects/UKB/IMAGING/subjectsAll34077/"
subject_data_dir = os.path.dirname(os.getcwd()) + "/real_data/"
GRF_data_dir = os.getcwd() + "/data/brain/"
HOMO_data_dir = os.getcwd() + "/data/brain/"

# Build filename components
def get_polynomial_suffix(polynomial_order):
    """Get polynomial order suffix for filenames."""
    if polynomial_order == 3:
        return "_cubic"
    elif polynomial_order == 1:
        return "_linear"
    return None

# Filename components
filename_components = {
    "dset": "_Simulation" if simulated_dset else "_RealDataset",
    "UKB_subject": f"UKB_{UKB_subject}" if not simulated_dset else "",
    "homo": "_Homogeneous" if homogeneous else "_BumpSignals",
    "model": "_full_model" if args.full_model else "_approximate_model",
    "poly": get_polynomial_suffix(polynomial_order),
    "firth": "_firth_penalty" if args.firth_penalty else "",
}

# Common parameters for UKB dataset
optimization_params = {
    "marginal": args.marginal_dist,
    "link": args.link_func,
    "gradient": args.gradient_mode,
    "precond": args.preconditioner_mode,
    "spacing": spacing,
}

# Helper function to ensure directory exists
def ensure_dir(filepath):
    """Create parent directories if they don't exist."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

# Helper function to build filenames
def build_filenames(simulated_dset, space_dim, model, args, filename_components, optimization_params, n_group, n_subject):
    """Build all necessary filenames based on dataset type and space dimension."""
    base_path = os.getcwd()
    
    if simulated_dset:
        if isinstance(space_dim, int):
            return _build_simulated_1d3d_filenames(base_path, space_dim, filename_components, optimization_params, 
                                                     n_group, n_subject, args)
        elif space_dim == "brain":
            return _build_simulated_brain_filenames(base_path, space_dim, model, filename_components, 
                                                     optimization_params, args)
        else:
            raise ValueError(f"Space dimension {space_dim} not supported")
    else:
        return _build_ukb_filenames(base_path, model, filename_components, optimization_params, args)

def _build_simulated_1d3d_filenames(base_path, space_dim, filename_components, optimization_params, n_group, n_subject, args):
    """Build filenames for 1D/2D/3D simulated data."""
    dim_str = f"{space_dim}D"
    common_suffix = f"{filename_components['homo']}{filename_components['model']}_{n_group}_group_{n_subject}_{optimization_params['marginal']}_{optimization_params['link']}_link_func"
    result_suffix = f"{common_suffix}_{optimization_params['gradient']}_{optimization_params['precond']}"
    
    filenames = {
        "data_filename": f"{base_path}/data/{dim_str}/{dim_str}_data{filename_components['dset']}{filename_components['homo']}_{n_group}_group_{n_subject}_{optimization_params['marginal']}_{optimization_params['link']}_link_func.npz",
        "results_filename": f"{base_path}/results/{dim_str}/{dim_str}_Regression{filename_components['dset']}{result_suffix}.npz",
        "inference_filename": f"{base_path}/inference/{dim_str}/{dim_str}_Inference_{args.inference_method}{filename_components['dset']}{common_suffix}.npz",
        "fig_filename": f"{base_path}/figures/PP-plots/{dim_str}/{dim_str}_PP_plot_{args.inference_method}{filename_components['dset']}{common_suffix}.png",
    }
    
    # Create directories
    for key in ["data_filename", "results_filename", "inference_filename"]:
        ensure_dir(filenames[key])
    
    return filenames

def _build_simulated_brain_filenames(base_path, space_dim, model, filename_components, optimization_params, args):
    """Build filenames for brain (spatial) simulated data."""
    model_params = f"{model}_{optimization_params['marginal']}_{optimization_params['link']}"
    common_suffix = f"{filename_components['dset']}{filename_components['model']}{filename_components['poly']}_{model}_{optimization_params['marginal']}_{optimization_params['link']}_link_func"
    result_suffix = f"{filename_components['dset']}{filename_components['model']}{filename_components['poly']}_{optimization_params['gradient']}_{optimization_params['precond']}_random_seed_{args.random_seed}"
    
    filenames = {
        "data_filename": f"{base_path}/data/{space_dim}/data{filename_components['dset']}/GRF_{args.n_subject}/GRF_{args.n_subject}_random_seed_{args.random_seed}.npz",
        "smooth_lesion_mask_filename": f"{base_path}/data/{space_dim}/smooth_lesion_mask{filename_components['dset']}.nii.gz",
        "results_filename": f"{base_path}/results/{space_dim}/GRF_{args.n_subject}/{model_params}/{space_dim}_Regression{result_suffix}.npz",
        "inference_filename": f"{base_path}/inference/{space_dim}/GRF_{args.n_subject}/{model_params}/{space_dim}_Inference_{args.inference_method}{filename_components['dset']}{filename_components['model']}{filename_components['poly']}_random_seed_{args.random_seed}.npz",
        "lesion_estimation_map_filename": f"{base_path}/results/{space_dim}/sqrt_P_mean{common_suffix}_random_seed_{args.random_seed}.png",
        "fig_filename": f"{base_path}/figures/PP-plots/{space_dim}/GRF_{args.n_subject}/{space_dim}_PP_plot_{args.inference_method}{common_suffix}_random_seed_{args.random_seed}.png",
    
    }
    
    # Create directories
    for key in ["results_filename", "inference_filename"]:
        ensure_dir(filenames[key])
    
    return filenames

def _build_ukb_filenames(base_path, model, filename_components, optimization_params, args):
    """Build filenames for UKB real dataset."""
    ukb_base_params = f"{model}{filename_components['dset']}{filename_components['model']}_{optimization_params['gradient']}_{optimization_params['precond']}_{optimization_params['marginal']}_{optimization_params['link']}_link_func_spacing_{optimization_params['spacing']}{filename_components['poly']}{filename_components['firth']}"
    common_suffix = f"{filename_components['dset']}{filename_components['model']}{filename_components['poly']}_{model}_{optimization_params['marginal']}_{optimization_params['link']}_link_func"
    
    lesion_mask = f"{base_path}/data/UKB/lesion_mask{filename_components['dset']}.nii.gz"
    filenames = {
        "data_filename": f"{base_path}/data/UKB/data{filename_components['dset']}.npz",
        "lesion_mask_filename": lesion_mask,
        "masked_data_filename": f"{base_path}/data/UKB/masked_data{filename_components['dset']}_spacing_{optimization_params['spacing']}.npz",
        "smooth_lesion_mask_filename": f"{base_path}/data/brain/smooth_lesion_mask_Simulation.nii.gz",
        "results_filename": f"{base_path}/results/{filename_components['UKB_subject']}/Regression_{ukb_base_params}.npz",
        "lesion_estimation_map_filename": f"{base_path}/figures/{filename_components['UKB_subject']}/spacing_{optimization_params['spacing']}/sqrt_P_mean{common_suffix}.png",
        "inference_filename": f"{base_path}/results/{filename_components['UKB_subject']}/Inference_{ukb_base_params}.npz",
        "XTWX_filename": f"{base_path}/results/{filename_components['UKB_subject']}/XTWX_{ukb_base_params}.npz",
        "Fisher_info_filename": f"{base_path}/results/{filename_components['UKB_subject']}/Fisher_info_{ukb_base_params}.npz",
        "meat_term_filename": f"{base_path}/results/{filename_components['UKB_subject']}/meat_term_{ukb_base_params}.npz",
        "bread_term_filename": f"{base_path}/results/{filename_components['UKB_subject']}/bread_term_{ukb_base_params}.npz",
        "p_vals_filename": f"{base_path}/results/{filename_components['UKB_subject']}/p_values_{ukb_base_params}_{args.inference_method}.npz",
        "z_vals_filename": f"{base_path}/results/{filename_components['UKB_subject']}/z_values_{ukb_base_params}_{args.inference_method}.npz",
        "fig_filename": f"{base_path}/figures/{filename_components['UKB_subject']}/spacing_{optimization_params['spacing']}/Z_map_{args.model}_{args.inference_method}{filename_components['dset']}{filename_components['model']}_{optimization_params['gradient']}_{optimization_params['precond']}_{optimization_params['marginal']}_{optimization_params['link']}_link_func_spacing_{optimization_params['spacing']}{filename_components['poly']}{filename_components['firth']}{args.inference_method}_{args.contrast_name}.png",
    }
    # Create directories
    for key in ["results_filename", "inference_filename", "fig_filename"]:
        ensure_dir(filenames[key])

    return filenames

# Generate filenames and unpack them into variables
filenames_dict = build_filenames(simulated_dset, space_dim, model, args, filename_components, 
                                  optimization_params, n_group, n_subject)

# Unpack all filenames (will vary based on dataset type)
data_filename = filenames_dict.get("data_filename")
results_filename = filenames_dict["results_filename"]
inference_filename = filenames_dict["inference_filename"]
p_vals_filename = filenames_dict.get("p_vals_filename")
z_vals_filename = filenames_dict.get("z_vals_filename")
fig_filename = filenames_dict["fig_filename"]
lesion_estimation_map_filename = filenames_dict.get("lesion_estimation_map_filename")
smooth_lesion_mask_filename = filenames_dict.get("smooth_lesion_mask_filename")
lesion_mask_filename = filenames_dict.get("lesion_mask_filename")
masked_data_filename = filenames_dict.get("masked_data_filename")
XTWX_filename = filenames_dict.get("XTWX_filename")
Fisher_info_filename = filenames_dict.get("Fisher_info_filename")
meat_term_filename = filenames_dict.get("meat_term_filename")
bread_term_filename = filenames_dict.get("bread_term_filename")

logging.info(f"load brain mask ...")
brain_mask_path = os.path.dirname(os.getcwd()) + "/GRF_data/MNI152_T1_2mm_brain_mask.nii.gz"
brain_mask = nib.load(brain_mask_path) if space_dim == "brain" else None
smooth_lesion_mask = nib.load(smooth_lesion_mask_filename) if smooth_lesion_mask_filename and os.path.exists(smooth_lesion_mask_filename) else None
n_voxels = np.sum(smooth_lesion_mask.get_fdata() > 0)

if args.run_data_generation:
    logging.info(f"Generate data{filename_components['dset']}...")
    # Check if spatial design matrix exists
    if simulated_dset:
        print(data_filename)
        if os.path.exists(data_filename):
            data = np.load(data_filename, allow_pickle=True)
            X_spatial = data[data.files[0]].item()["X_spatial"]
        else:
            brain_mask = None if isinstance(space_dim, int) else brain_mask
            # create data file 
            os.makedirs(os.path.dirname(data_filename), exist_ok=True)
            # X_spatial = B_spline_bases(space_dim=space_dim, dim=n_voxels, brain_mask=smooth_lesion_mask, spacing=spacing, dtype=np.float64)
            X_spatial = QMCFeatures_3D(brain_mask=smooth_lesion_mask, length_scale=1.0, n_features=445)
            np.savez(data_filename, X_spatial=X_spatial)
    else: 
        if os.path.exists(smooth_lesion_mask_filename):
            smooth_lesion_mask = nib.load(smooth_lesion_mask_filename)
            # X_spatial = QMCFeatures_3D(brain_mask=smooth_lesion_mask, length_scale=1.0, n_features=445)
            # X_spatial = RandomFourierFeatures_3D(space_dim=space_dim, dim=n_voxels, brain_mask=smooth_lesion_mask, n_features=800, sigma=0.1)
            X_spatial = B_spline_bases(space_dim=space_dim, dim=n_voxels, brain_mask=smooth_lesion_mask, spacing=spacing, dtype=np.float64)
    if simulated_dset:
        if isinstance(space_dim, int):
            lesion_size_mapping = {
                1: [1, 3],
                2: [1, 8],
                3: [1, 16],
                "brain": [1, 16]
            }
            lesion_size_range = lesion_size_mapping.get(space_dim, None)

            data_simulation = simulated_data(space_dim=space_dim, n_group=n_group, n_subject=n_subject, n_voxel=n_voxels, brain_mask=brain_mask,
                                            group_names=group_names, homogeneous_intensity=homogeneous, lesion_per_subject=lesion_per_subject)
            G, MU, Y, Z = data_simulation.generate_data(lesion_size_range=lesion_size_range)
            data = dict(G=G, MU=MU, X_spatial=X_spatial, Y=Y, Z=Z)
            np.savez(data_filename, **data)
        elif space_dim == "brain":
            # pre_processed_data = SubjectHomo_simulated_data(n_group=n_group, n_subject=n_subject, HOMO_data_dir=HOMO_data_dir)
            # Z, Y = pre_processed_data.process_data(mask_path=smooth_lesion_mask_filename, random_seed=args.random_seed)
            # pre_processed_data = SpatialHomo_simulated_data(n_group=n_group, n_subject=n_subject, HOMO_data_dir=HOMO_data_dir)
            # Z, Y = pre_processed_data.process_data(mask_path=smooth_lesion_mask_filename, random_seed=args.random_seed)
            pre_processed_data = GRF_simulated_data(GRF_data_dir=GRF_data_dir, n_group=n_group, n_subject=n_subject, group_names=group_names)
            data = pre_processed_data.process_data(mask_path=smooth_lesion_mask_filename, random_seed=args.random_seed)
            # Add X_spatial to each group dictionary
            {data[group_name].update({"X_spatial": X_spatial}) for group_name in group_names}
            np.savez(data_filename, **data)
    else:
        if not os.path.isfile(data_filename):
            brain_mask_path = os.path.join(subject_data_dir, f"MNI152_T1_2mm_brain_mask.nii.gz")
            pre_processed_data = Biobank_data(data_dir, subject_data_dir)
            Z, Y = pre_processed_data.process_data(brain_mask_path)
            data = dict(Y=Y, Z=Z)
            np.savez(data_filename, **data)
        if not os.path.isfile(lesion_mask_filename):
            data = np.load(data_filename, allow_pickle=True) if "data" not in locals() else data
            p_empirical = data["Y"].mean(axis=0)
            create_lesion_mask(p_empirical, brain_mask, lesion_mask_filename, threshold=1e-3)
        if not os.path.isfile(masked_data_filename):
            masked_data = Biobank_data(data_dir, subject_data_dir)
            Z, Y = masked_data.process_data(smooth_lesion_mask_filename)
            data = dict(X_spatial=X_spatial, Y=Y, Z=Z)
            np.savez(masked_data_filename, **data)
        exit()
        # data = np.load(masked_data_filename, allow_pickle=True)
        # data = {key: data[key] for key in data.files}
        # print(data["X_spatial"])
        # print(data["X_spatial"].shape)
        # data["X_spatial"] = X_spatial
        # print(data["X_spatial"].shape)
        # np.savez(masked_data_filename, **data)
        # exit()

if args.run_regression:
    logging.info("Setup model and optimise regression coefficients")
    if not "data" in locals():
        if simulated_dset:
            data = np.load(data_filename, allow_pickle=True)
        else:
            data = np.load(masked_data_filename, allow_pickle=True)
        data = {key: data[key] for key in data.files}
    if not simulated_dset and space_dim == "brain":
        #######################
        # subset data for testing model performance
        if UKB_subject < n_subjects_whole_UKB:
            np.random.seed(42)
            selected_indices = np.random.choice(n_subjects_whole_UKB, size=UKB_subject, replace=False)
            # total_needed = 5 * UKB_subject
            # all_selected_indices = np.random.choice(n_subjects_whole_UKB, size=total_needed, replace=False)
            # selected_indices = all_selected_indices[4*UKB_subject:]
            data["Y"] = data["Y"][selected_indices]
            data["Z"] = data["Z"][selected_indices]
        #######################
    # add cubic terms to Z
    for group_name in group_names:
        data[group_name].item()["Z"] = preprocess_Z(simulated_dset, data[group_name].item()["Z"], polynomial_order)
    result = {}
    if args.full_model:
        if not os.path.exists(results_filename):
            BR = BrainRegression_full(dtype=torch.float64, device=device)
            BR.load_data(data)
            BR.init_model(model, 
                        n_auxiliary=args.n_auxiliary, 
                        std_auxiliary=args.std_auxiliary,
                        n_samples=args.n_samples,
                        regression_terms=args.regression_terms,
                        link_func=args.link_func,
                        marginal_dist=args.marginal_dist,
                        std_params=args.std_params,
                        firth_penalty=args.firth_penalty)
            
            print("Optimising regression coefficients ...")
            start_time = time.time()
            BR.optimize_model(lr, iter, tolerance_change)
            print(f"Optimization time: {time.time() - start_time} seconds")
            # save optimised params
            if hasattr(BR.model, 'betas') and BR.model.betas is not None:
                beta = {g: BR.model.betas[g].detach().cpu().numpy() for g in BR.model.betas}
            else:
                beta = BR.model.beta.detach().cpu().numpy()
            MU_dict = BR.model(BR.B, BR.Y, BR.Z)
            if isinstance(MU_dict, dict):
                MU_mean = {g: mu.detach().cpu().numpy().mean(axis=0) for g, mu in MU_dict.items()}
                MU_std = {g: mu.detach().cpu().numpy().std(axis=0) for g, mu in MU_dict.items()}
                MU_np = {g: mu.detach().cpu().numpy() for g, mu in MU_dict.items()}
                P = {g: MU_np[g] * np.exp(-MU_np[g]) for g in MU_np}
            else:
                MU_np = MU_dict.detach().cpu().numpy()
                MU_mean = MU_np.mean(axis=0)
                MU_std = MU_np.std(axis=0)
                P = MU_np * np.exp(-MU_np)
            result = {"beta": beta, "P": P, "MU_mean": MU_mean, "MU_std": MU_std}
            print(results_filename)
            np.savez(results_filename, **result)
        else:
            print(results_filename)
            logging.info(f"Results file {results_filename} already exists. Skipping regression.")
            beta = np.load(results_filename, allow_pickle=True)["beta"]
            P_raw = np.load(results_filename, allow_pickle=True)["P"]
            P_dict = P_raw.item() if P_raw.ndim == 0 else P_raw
            if isinstance(P_dict, dict):
                first_group = list(P_dict.keys())[0]
                P_mean = np.mean(P_dict[first_group], axis=0)
            else:
                P_mean = np.mean(P_dict, axis=0)
            plot_brain(p=np.sqrt(P_mean), brain_mask=smooth_lesion_mask, vmax=None, output_filename=os.getcwd() + f"/test.png")
    else:
        print("herehere")
        print(results_filename)
        print(os.path.exists(results_filename))
        dask.config.set({"distributed.worker.nthreads": 1})  # Threads per worker
        dask.config.set({"distributed.workers": os.cpu_count()})  # Number of workers
        logging.set_verbosity(logging.INFO)
        start_time = time.time()
        BR = BrainRegression_Approximate(simulated_dset=simulated_dset, dtype=torch.float64, device=device)
        BR.load_data(data, model)
        if not os.path.exists(results_filename):
            alpha = 0.01
            start_time = time.time()
            beta = BR.run_regression(model=model,
                                    marginal_dist=args.marginal_dist,
                                    link_func=args.link_func,
                                    max_iter=1000, 
                                    alpha = alpha,
                                    gradient_mode=args.gradient_mode,
                                    preconditioner_mode=args.preconditioner_mode,
                                    block_size=5000,
                                    compute_nll=True)
            print(f"Regression optimization time: {time.time() - start_time} seconds")
            MU_mean, MU_std, P_mean = BR.goodness_of_fit(beta=beta, model=model, mode="dask", block_size=5000)
            print('result_name:', results_filename)
            print('fig_name:', filenames_dict['lesion_estimation_map_filename'])
            print(smooth_lesion_mask is None)
            np.savez(results_filename, beta=beta, MU_mean=MU_mean, MU_std=MU_std, P_mean=P_mean)
            plot_brain(np.sqrt(P_mean), smooth_lesion_mask, vmax=None, threshold=0, output_filename=filenames_dict['lesion_estimation_map_filename']) if smooth_lesion_mask is not None else plot_brain(np.sqrt(P_mean), brain_mask, output_filename=filenames_dict['lesion_estimation_map_filename'])
            save_nifti(P_mean, smooth_lesion_mask, filenames_dict['lesion_estimation_map_filename'].replace('.png', '.nii.gz')) if smooth_lesion_mask is not None else save_nifti(P_mean, brain_mask, filenames_dict['lesion_estimation_map_filename'].replace('.png', '.nii.gz'))
    # if args.full_model:
    #     import matplotlib.pyplot as plt
    #     from plot import plot_intensity_1d, plot_intensity_2d, plot_intensity_3d, plot_brain
    #     fig_filename = f"{os.getcwd()}/figures/probability_maps/{space_dim}D/{space_dim}D_Probability_comparison{filename_0}{filename_1}_{n_group}_group_{args.marginal_dist}_{args.link_func}_link_func.png"
    #     print(fig_filename)
    #     if space_dim == 1:
    #         plot_intensity_1d(G, MU, P, fig_filename)
    #     elif space_dim == 2:
    #         plot_intensity_2d(G, MU, P, n_voxel, fig_filename)
    #     elif space_dim == 3:
    #         plot_intensity_3d(G, MU, P, n_voxel, fig_filename)
    #     elif space_dim == "brain":
    #         P_mean = np.mean(P, axis=0) # average over subjects
    #         plot_brain(p=np.sqrt(P_mean), brain_mask=smooth_lesion_mask, output_filename=lesion_estimation_map_filename) 
    
if args.run_inference:
    logging.info("Conduct statistical inference via either Fisher Information or Sandwich estimator")
    if not "data" in locals():
        if simulated_dset:
            data = np.load(data_filename, allow_pickle=True)
        else:
            data = np.load(masked_data_filename, allow_pickle=True)
        data = {key: data[key] for key in data.files}
    if not simulated_dset and space_dim == "brain":
        #######################
        # subset data for testing model performance
        if UKB_subject < n_subjects_whole_UKB:
            np.random.seed(42)
            selected_indices = np.random.choice(n_subjects_whole_UKB, size=UKB_subject, replace=False)
            # total_needed = 5 * UKB_subject
            # all_selected_indices = np.random.choice(n_subjects_whole_UKB, size=total_needed, replace=False)
            # selected_indices = all_selected_indices[4*UKB_subject:]
            data["Y"] = data["Y"][selected_indices]
            data["Z"] = data["Z"][selected_indices]
        #######################
    # add cubic terms to Z
    for group_name in group_names:
        data[group_name].item()["Z"] = preprocess_Z(simulated_dset, data[group_name].item()["Z"], polynomial_order)

    # load optimised params
    print("results: ", results_filename)
    results = np.load(results_filename, allow_pickle=True)
    if args.full_model:
        # BrainInference
        BI = BrainInference_full(model=model, space_dim=space_dim,marginal_dist=args.marginal_dist, 
                            link_func=args.link_func, regression_terms=args.regression_terms,
                            random_seed=args.random_seed, fewer_voxels=False,dtype=torch.float64, device=device)
        BI.load_params(data=data, params=results)
        BI.create_contrast(contrast_vector=args.contrast_vector, contrast_name=args.contrast_name)
        BI.run_inference(method=args.inference_method, inference_filename=inference_filename,
                         fig_filename=fig_filename, lesion_mask=smooth_lesion_mask)
    else:
        if simulated_dset:
            BI = BrainInference_Approximate(model=model, marginal_dist=args.marginal_dist, 
                                link_func=args.link_func, regression_terms=args.regression_terms,
                                dtype=torch.float64, device=device)
        else:
            BI = BrainInference_UKB(model=model, marginal_dist=args.marginal_dist, 
                                link_func=args.link_func, regression_terms=args.regression_terms,
                                dtype=torch.float64, device=device)
        BI.load_params(data=data, params=results)
        BI.create_contrast(contrast_vector=args.contrast_vector, contrast_name=args.contrast_name,
                           polynomial_order=polynomial_order)
        if simulated_dset:
            BI.run_inference(method=args.inference_method, inference_filename=inference_filename, fig_filename=fig_filename)
        else:
            BI.run_inference(method=args.inference_method, lesion_mask=smooth_lesion_mask, 
                            XTWX_filename=XTWX_filename, Fisher_info_filename=Fisher_info_filename,
                            meat_term_filename=meat_term_filename, bread_term_filename=bread_term_filename,
                            p_vals_filename=p_vals_filename, z_vals_filename=z_vals_filename,
                            fig_filename=fig_filename)