import numpy as np
from util import preprocess_Z, SpatialGLM_compute_P_mean
from plot import plot_brain, save_nifti
import nibabel as nib
import os
import matplotlib.pyplot as plt

polynomial_order = 1
filename = "linear" if polynomial_order == 1 else "cubic" if polynomial_order == 3 else None
simulated_dset = False
spacing = 5
N_UKB_subjects = 13677

brain_mask = nib.load("/well/nichols/users/pra123/brain_lesion_project/real_data/MNI152_T1_2mm_brain_mask.nii.gz")
smooth_lesion_mask = nib.load("data/UKB/smooth_lesion_mask_RealDataset.nii.gz")

data = np.load(f"data/UKB/masked_data_RealDataset_spacing_{spacing}.npz")
Z, B = data["Z"], data["X_spatial"]
Z = preprocess_Z(simulated_dset, Z, polynomial_order)
Z = Z * 50 / Z.shape[0]
B = B * 50 / B.shape[0]
Z = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
B = np.concatenate([B, np.ones((B.shape[0], 1))], axis=1)
Z_mean = np.mean(Z, axis=0)
Z_age_5 = np.array([0, 5, 0, 0, 0], dtype=np.float64).reshape((1,-1))
Z_cvr_half = np.array([0, 0, 0, 0.5, 0], dtype=np.float64).reshape((1,-1))
Z_age_5 *= 50 / Z.shape[0]
Z_cvr_half *= 50 / Z.shape[0]

# MUM: Mass Univariate Regression
MUM_results = np.load(f"results/UKB_{N_UKB_subjects}/Regression_MassUnivariateRegression_RealDataset_approximate_model_dask_approximate_Poisson_log_link_func_spacing_5_linear.npz", allow_pickle=True)
MUM_beta, P_MUM_mean = MUM_results["beta"], MUM_results["P_mean"]
if not os.path.exists(f"RR_maps_{N_UKB_subjects}/P_MUM_mean.png"):
    plot_brain(P_MUM_mean, smooth_lesion_mask, slice_idx=None, threshold=0, output_filename=f"RR_maps_{N_UKB_subjects}/P_MUM_mean.png")
    save_nifti(P_MUM_mean.flatten(), smooth_lesion_mask, f"RR_maps_{N_UKB_subjects}/P_MUM_mean.nii.gz")

mu_MUM_age_plus_5 = np.exp((Z+Z_age_5) @ MUM_beta)
P_MUM_age_plus_5 = mu_MUM_age_plus_5 * np.exp(-mu_MUM_age_plus_5)
mu_MUM_age_minus_5 = np.exp((Z-Z_age_5) @ MUM_beta)
P_MUM_age_minus_5 = mu_MUM_age_minus_5 * np.exp(-mu_MUM_age_minus_5)
P_MUM_age_plus_5_mean = np.mean(P_MUM_age_plus_5, axis=0)
P_MUM_age_minus_5_mean = np.mean(P_MUM_age_minus_5, axis=0)
RR_MUM_age = P_MUM_age_plus_5_mean / P_MUM_age_minus_5_mean
RD_MUM_age = P_MUM_age_plus_5_mean - P_MUM_age_minus_5_mean
if not os.path.exists(f"RR_maps_{N_UKB_subjects}/RR_MUM_age.png"):
    plot_brain(RR_MUM_age, smooth_lesion_mask, slice_idx=None, threshold=0, output_filename=f"RR_maps_{N_UKB_subjects}/RR_MUM_age.png")
    save_nifti(RR_MUM_age.flatten(), smooth_lesion_mask, f"RR_maps_{N_UKB_subjects}/RR_MUM_age.nii.gz")
if not os.path.exists(f"RR_maps_{N_UKB_subjects}/RD_MUM_age.png"):
    plot_brain(RD_MUM_age, smooth_lesion_mask, slice_idx=None, threshold=0, output_filename=f"RR_maps_{N_UKB_subjects}/RD_MUM_age.png")
    save_nifti(RD_MUM_age.flatten(), smooth_lesion_mask, f"RR_maps_{N_UKB_subjects}/RD_MUM_age.nii.gz")
if not os.path.exists(f"RR_maps_{N_UKB_subjects}/Scatter_plot_MUM_age.png"):
    plt.figure()
    sc = plt.scatter(RR_MUM_age, RD_MUM_age, s=0.5)
    plt.xlabel("RR")
    plt.ylabel("RD")
    plt.title("RR vs RD for MUM-Age")
    plt.savefig(f"RR_maps_{N_UKB_subjects}/Scatter_plot_MUM_age.png")

mu_MUM_cvr_plus_half = np.exp((Z + Z_cvr_half) @ MUM_beta)
P_MUM_cvr_plus_half = mu_MUM_cvr_plus_half * np.exp(-mu_MUM_cvr_plus_half)
P_MUM_cvr_plus_half_mean = np.mean(P_MUM_cvr_plus_half, axis=0)
mu_MUM_cvr_minus_half = np.exp((Z - Z_cvr_half) @ MUM_beta)
P_MUM_cvr_minus_half = mu_MUM_cvr_minus_half * np.exp(-mu_MUM_cvr_minus_half)
P_MUM_cvr_minus_half_mean = np.mean(P_MUM_cvr_minus_half, axis=0)
RR_MUM_cvr = P_MUM_cvr_plus_half_mean / P_MUM_cvr_minus_half_mean
RD_MUM_cvr = P_MUM_cvr_plus_half_mean - P_MUM_cvr_minus_half_mean
if not os.path.exists("RR_maps/RR_MUM_cvr.png"):
    plot_brain(RR_MUM_cvr, smooth_lesion_mask, slice_idx=None, threshold=0, output_filename="RR_maps/RR_MUM_cvr.png")
    save_nifti(RR_MUM_cvr.flatten(), smooth_lesion_mask, "RR_maps/RR_MUM_cvr.nii.gz")
if not os.path.exists("RR_maps/RD_MUM_cvr.png"):
    plot_brain(RD_MUM_cvr, smooth_lesion_mask, slice_idx=None, threshold=0, output_filename="RR_maps/RD_MUM_cvr.png")
    save_nifti(RD_MUM_cvr.flatten(), smooth_lesion_mask, "RR_maps/RD_MUM_cvr.nii.gz")
print(RR_MUM_cvr.shape, RD_MUM_cvr.shape)
if not os.path.exists("RR_maps/Scatter_plot_MUM_cvr.png"):
    plt.figure()
    sc = plt.scatter(RR_MUM_cvr, RD_MUM_cvr, s=0.5)
    plt.xlabel("RR")
    plt.ylabel("RD")
    plt.title("RR vs RD for MUM-CVR")
    plt.savefig("RR_maps/Scatter_plot_MUM_cvr.png")



# S-GLM
SGLM_results = np.load(f"results/UKB_{N_UKB_subjects}/Regression_SpatialBrainLesion_RealDataset_approximate_model_dask_approximate_Poisson_log_link_func_spacing_5_linear.npz", allow_pickle=True)
beta, MU_mean, P_mean = SGLM_results["beta"], SGLM_results["MU_mean"], SGLM_results["P_mean"]
if not os.path.exists("RR_maps/P_SGLM_mean.png"):
    plot_brain(P_mean, smooth_lesion_mask, slice_idx=None, threshold=0, output_filename="RR_maps/P_SGLM_mean.png")
    save_nifti(P_mean.flatten(), smooth_lesion_mask, "RR_maps/P_SGLM_mean.nii.gz")

eta_age_plus_5 = np.kron((Z_mean + Z_age_5), B) @ beta
eta_age_minus_5 = np.kron((Z_mean - Z_age_5), B) @ beta
mu_age_plus_5 = np.exp(eta_age_plus_5)
mu_age_minus_5 = np.exp(eta_age_minus_5)
P_age_plus_5 = mu_age_plus_5 * np.exp(-mu_age_plus_5)
P_age_minus_5 = mu_age_minus_5 * np.exp(-mu_age_minus_5)
RR_SGLM_age = mu_age_plus_5 / mu_age_minus_5
RD_SGLM_age = P_age_plus_5 - P_age_minus_5
if not os.path.exists("RR_maps/RR_SGLM_age.png"):
    plot_brain(RR_SGLM_age, smooth_lesion_mask, slice_idx=None, threshold=1, output_filename="RR_maps/RR_SGLM_age.png")
    save_nifti(RR_SGLM_age.flatten(), smooth_lesion_mask, "RR_maps/RR_SGLM_age.nii.gz")
if not os.path.exists("RR_maps/RD_SGLM_age.png"):
    plot_brain(RD_SGLM_age, smooth_lesion_mask, slice_idx=None, threshold=0, output_filename="RR_maps/RD_SGLM_age.png")
    save_nifti(RD_SGLM_age.flatten(), smooth_lesion_mask, "RR_maps/RD_SGLM_age.nii.gz")
if not os.path.exists("RR_maps/Scatter_plot_SGLM_age.png"):
    plt.figure()
    sc = plt.scatter(RR_SGLM_age, RD_SGLM_age, s=0.5)
    plt.xlabel("RR")
    plt.ylabel("RD")
    plt.title("RR vs RD for SGLM-Age")
    plt.savefig("RR_maps/Scatter_plot_SGLM_age.png")

eta_cvr_plus_half = np.kron(Z_mean + Z_cvr_half, B) @ beta
eta_cvr_minus_half = np.kron(Z_mean - Z_cvr_half, B) @ beta
mu_cvr_plus_half = np.exp(eta_cvr_plus_half)
mu_cvr_minus_half = np.exp(eta_cvr_minus_half)
P_cvr_plus_half = mu_cvr_plus_half * np.exp(-mu_cvr_plus_half)
P_cvr_minus_half = mu_cvr_minus_half * np.exp(-mu_cvr_minus_half)
RR_SGLM_cvr = mu_cvr_plus_half / mu_cvr_minus_half
RD_SGLM_cvr = P_cvr_plus_half - P_cvr_minus_half
if not os.path.exists("RR_maps/RR_SGLM_cvr.png"):
    plot_brain(RR_SGLM_cvr, smooth_lesion_mask, slice_idx=None, threshold=1, output_filename="RR_maps/RR_SGLM_cvr.png")
    save_nifti(RR_SGLM_cvr.flatten(), smooth_lesion_mask, "RR_maps/RR_SGLM_cvr.nii.gz")
if not os.path.exists("RR_maps/RD_SGLM_cvr.png"):
    plot_brain(RD_SGLM_cvr, smooth_lesion_mask, slice_idx=None, threshold=0, output_filename="RR_maps/RD_SGLM_cvr.png")
    save_nifti(RD_SGLM_cvr.flatten(), smooth_lesion_mask, "RR_maps/RD_SGLM_cvr.nii.gz")
if not os.path.exists("RR_maps/Scatter_plot_SGLM_cvr.png"):
    plt.figure()
    sc = plt.scatter(RR_SGLM_cvr, RD_SGLM_cvr, s=0.5)
    plt.xlabel("RR")
    plt.ylabel("RD")
    plt.title("RR vs RD for SGLM-CVR")
    plt.savefig("RR_maps/Scatter_plot_SGLM_cvr.png")