import numpy as np
import nibabel as nib
import scipy.stats
from plot import plot_brain
import os

smooth_brain_mask_path = "data/brain/smooth_lesion_mask_Simulation.nii.gz"
smooth_lesion_mask = nib.load(smooth_brain_mask_path)
mask = smooth_lesion_mask.get_fdata().astype(bool)

def load_masked_coef(filename):
    """Load a NIfTI coefficient map and apply the lesion mask."""
    return nib.load(filename).get_fdata().astype(np.float32)[mask]

coef_age_ukbb = load_masked_coef("data/brain/coef_age_nvars_1_method_2.nii.gz")
coef_intercept_ukbb = load_masked_coef("data/brain/coef_Intercept_nvars_1_method_2.nii.gz")
empir_prob = load_masked_coef("data/brain/empir_prob_mask.nii.gz")

# Ground truth P = Phi(intercept), zeroed where empirical prob is 0
ground_truth_P = scipy.stats.norm.cdf(coef_intercept_ukbb)
ground_truth_P[empir_prob == 0] = 0.0
ground_truth_P = ground_truth_P.reshape(1, -1) # shape: (1, 14807)

results = []
for i in range(100):
    result_i = np.load("results/brain/GRF_[100]/SpatialBrainLesion_Poisson_log/brain_Regression_Simulation_full_model_linear_random_seed_0.npz", allow_pickle=True)["P"].item()["Group_1"]
    print(result_i.shape) # shape: (14807,)
    a = np.mean(result_i, axis=0)
    print(np.mean(a-ground_truth_P))
    # plot_brain(a, brain_mask=smooth_lesion_mask, vmax=None, output_filename=os.getcwd() + f"/test.png")
    exit()
    results.append(result_i)
results = np.stack(results) # shape: (100, 14807)

bias = np.mean(results - ground_truth_P)
std = np.std(results - ground_truth_P)
MSE = np.mean((results - ground_truth_P)**2)
print(bias, std, MSE)

relative_bias = bias / np.mean(ground_truth_P)
relative_std = std / np.mean(ground_truth_P)
relative_MSE = MSE / np.mean(ground_truth_P**2)
print(relative_bias, relative_std, relative_MSE)

# bias = np.mean(ground_truth_P - results, axis=0)
# std = np.std(ground_truth_P - results, axis=0)
# MSE = np.mean((ground_truth_P - results)**2, axis=0)
# print(bias, std, MSE)

# relative_bias = bias / np.mean(ground_truth_P)
# relative_std = std / np.mean(ground_truth_P)
# relative_MSE = MSE / np.mean(ground_truth_P**2)
# print(relative_bias, relative_std, relative_MSE)