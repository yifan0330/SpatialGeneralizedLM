import numpy as np
import nibabel as nib
import scipy.stats
from plot import plot_brain
import os

model = "SpatialBrainLesion" # "MassUnivariateRegression"
distribution = "NB" #"Poisson"
inference = "sandwich" #"FI"

smooth_brain_mask_path = "data/brain/smooth_lesion_mask_Simulation.nii.gz"
smooth_lesion_mask = nib.load(smooth_brain_mask_path)
mask = smooth_lesion_mask.get_fdata().astype(bool)

def load_masked_coef(filename):
    """Load a NIfTI coefficient map and apply the lesion mask."""
    return nib.load(filename).get_fdata().astype(np.float32)[mask]

coef_age_ukbb = load_masked_coef("data/brain/coef_age_nvars_1_method_2.nii.gz")
coef_intercept_ukbb = load_masked_coef("data/brain/coef_Intercept_nvars_1_method_2.nii.gz")
empir_prob = load_masked_coef("data/brain/empir_prob_mask.nii.gz")

# Ground truth: empirical mean of Y across all seeds and subjects
Y_all = []
for i in range(100):
    Y_i = np.load(f"data/brain/data_Simulation/GRF_[100]/GRF_[100]_random_seed_{i}.npz",
                  allow_pickle=True)["Group_1"].item()["Y"]  # (100, 14807)
    Y_all.append(Y_i.mean(axis=0))  # average across subjects → (14807,)
ground_truth_MU = np.mean(np.stack(Y_all), axis=0).reshape(1, -1)  # average across seeds → (1, 14807)

# Y_approx_approx = np.load("results/brain/GRF_[100]/SpatialBrainLesion_Poisson_log/brain_Regression_Simulation_approximate_model_linear_approximate_approximate_random_seed_0.npz", allow_pickle=True)["MU_mean"]
# Y_approx_approx = Y_approx_approx.reshape(1, -1)  # (1, n_voxel)
# bias = np.mean(Y_approx_approx - ground_truth_MU)        # (14807,) mean across seeds
# std = np.std(Y_approx_approx - ground_truth_MU)          # (14807,)
# MSE = np.mean((Y_approx_approx - ground_truth_MU)**2)    # (14807,)
# # rel
# rel_bias = bias / np.mean(np.abs(ground_truth_MU))
# rel_std = std / np.mean(np.abs(ground_truth_MU))
# rel_MSE = MSE / np.mean(ground_truth_MU**2)

Y_dask_approx = np.load("results/brain/GRF_[100]/SpatialBrainLesion_Poisson_log/brain_Regression_Simulation_approximate_model_linear_dask_approximate_random_seed_0.npz", allow_pickle=True)["MU_mean"]
Y_dask_approx = Y_dask_approx.reshape(1, -1)  # (1, n_voxel)
# compute relative bias, std, MSE, compared to ground truth, and average across voxels
errors = Y_dask_approx - ground_truth_MU
bias = np.mean(errors)
std  = np.std(errors)
MSE  = np.mean(errors**2)

scale = np.mean(np.abs(ground_truth_MU))
rel_bias = bias / scale
rel_std  = std / scale
rel_MSE  = MSE / np.mean(ground_truth_MU**2)
print(rel_bias, rel_std, rel_MSE)
exit()



results = []
for i in range(100):
    result_i = np.load(f"results/brain/GRF_[1000]/{model}_{distribution}_log/brain_Regression_Simulation_full_model_linear_random_seed_{i}.npz", allow_pickle=True)["MU_mean"].item()["Group_1"]
    # plot_brain(result_i, brain_mask=smooth_lesion_mask, vmax=None,
    #            output_filename=os.getcwd() + "/test.png")
    # result_i = np.load(f"results/brain/GRF_[1000]/MassUnivariateRegression_Poisson_log/brain_Regression_Simulation_full_model_linear_random_seed_{i}.npz", allow_pickle=True)["MU_mean"].item()["Group_1"]
    results.append(result_i)
results = np.stack(results) # shape: (100, 14807)
print(results.shape, ground_truth_MU.shape)
# remove rows (seeds) that contain any NaN
valid_rows = ~np.isnan(results).any(axis=1)
results = results[valid_rows, :]

bias = np.mean(results - ground_truth_MU, axis=0)        # (14807,) mean across seeds
std = np.std(results - ground_truth_MU, axis=0)          # (14807,)
MSE = np.mean((results - ground_truth_MU)**2, axis=0)    # (14807,)
print("Per-voxel bias: mean={:.10f}, median={:.10f}".format(np.mean(bias), np.median(bias)))
print("Per-voxel std:  mean={:.10f}, median={:.10f}".format(np.mean(std), np.median(std)))
print("Per-voxel MSE:  mean={:.10f}, median={:.10f}".format(np.mean(MSE), np.median(MSE)))


# Plot brain maps
plot_brain(bias, brain_mask=smooth_lesion_mask, vmax=None,
           output_filename=os.getcwd() + "/figures/bias_map.png")
plot_brain(std, brain_mask=smooth_lesion_mask, vmax=None,
           output_filename=os.getcwd() + "/figures/std_map.png")
plot_brain(MSE, brain_mask=smooth_lesion_mask, vmax=None,
           output_filename=os.getcwd() + "/figures/MSE_map.png")