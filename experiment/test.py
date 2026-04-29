import numpy as np
import matplotlib.pyplot as plt
import os

cd /well/nichols/users/pra123/brain_lesion_project/experiment

python - <<'EOF'
import numpy as np
from util import preprocess_Z
from plot import save_nifti
import nibabel as nib

polynomial_order = 1
spacing = 5
N_UKB_subjects = 13677

smooth_lesion_mask = nib.load("data/UKB/smooth_lesion_mask_RealDataset.nii.gz")

data = np.load(f"data/UKB/masked_data_RealDataset_spacing_{spacing}.npz")
Z, B = data["Z"], data["X_spatial"]
Z = preprocess_Z(False, Z, polynomial_order)
Z = Z * 50 / Z.shape[0]
Z = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
Z_age_5 = np.array([0, 5, 0, 0, 0], dtype=np.float64).reshape((1,-1)) * 50 / Z.shape[0]

MUM_results = np.load(
    f"results/UKB_{N_UKB_subjects}/Regression_MassUnivariateRegression_RealDataset_approximate_model_dask_approximate_Poisson_log_link_func_spacing_5_linear.npz",
    allow_pickle=True)
MUM_beta = MUM_results["beta"]

mu_plus  = np.exp((Z + Z_age_5) @ MUM_beta)
mu_minus = np.exp((Z - Z_age_5) @ MUM_beta)
P_plus   = mu_plus  * np.exp(-mu_plus)
P_minus  = mu_minus * np.exp(-mu_minus)
RD_MUM_age = np.mean(P_plus, axis=0) - np.mean(P_minus, axis=0)
print("RD_MUM_age shape:", RD_MUM_age.shape)
exit()

save_nifti(RD_MUM_age.flatten(), smooth_lesion_mask, "RR_maps_13677/RD_MUM_age.nii.gz")
print("Saved RR_maps_13677/RD_MUM_age.nii.gz")
exit()


result_path = "results/UKB_13677/Regression_MassUnivariateRegression_RealDataset_approximate_model_dask_approximate_Poisson_log_link_func_spacing_5_linear.npz"
results = np.load(result_path, allow_pickle=True)
beta = results["beta"]
beta = beta.reshape(5, -1)
print(beta[1,:])
exit()


data_path = "/well/nichols/users/pra123/brain_lesion_project/experiment/data/UKB/masked_data_RealDataset_spacing_5.npz"
data = np.load(data_path, allow_pickle=True)
data_dict = {key: data[key] for key in data.keys()}
Y = data_dict["Y"]
empirical_P = np.mean(Y, axis=0) 

RFF_features = [100, 200, 400, 800]

for n in RFF_features:
    file_path = f"/well/nichols/users/pra123/brain_lesion_project/experiment/results/UKB_13677_RFF{n}/Regression_SpatialBrainLesion_RealDataset_approximate_model_dask_approximate_Poisson_log_link_func_spacing_5_linear.npz"
    npz_file = np.load(file_path, allow_pickle=True)
    # Method 3: Load all data into a dictionary
    regression_dict = {key: npz_file[key] for key in npz_file.keys()}
    P_mean = regression_dict["P_mean"]
    bias_P = P_mean - empirical_P
    print("number of fourier features:", n, "mean bias is:", np.mean(bias_P))
exit()


print(os.getcwd()+"svdvals.npz")
s = np.load(os.getcwd()+"svdvals.npz")["s"]
# Sort the array in descending order
sorted_data = np.sort(s)[::-1]

print(np.max(sorted_data)/np.min(sorted_data))
exit()
# Plot the sorted values
plt.figure(figsize=(10, 5))
plt.plot(sorted_data[5000:], marker='o')  # Line plot with points
plt.xlabel("Index (sorted)")
plt.ylabel("Value")
plt.title("Sorted singular values in Decreasing Order")
plt.grid(True)
plt.savefig(os.getcwd()+"/svdvals.png")
exit()


results_bernoulli = np.load("results/1D_Probability_comparison_Simulation_Homogeneous_1_group_Bernoulli_logit_link_func.npz", allow_pickle=True)
P_bernoulli = results_bernoulli["P"]

results_poission = np.load("results/1D_Probability_comparison_Simulation_Homogeneous_1_group_Poisson_log_link_func.npz", allow_pickle=True)
P_poission = results_poission["P"]

P_diff = P_bernoulli - P_poission

n_subject = P_bernoulli.shape[0]

q1 = 0
q2 = int(np.percentile(np.arange(n_subject), 25))
q3 = int(np.percentile(np.arange(n_subject), 75))
q4 = n_subject - 1

# Bias plot
fig, axs = plt.subplots(2, 2, figsize=(10, 11))
# Subplot 1
axs[0, 0].plot(P_diff[q1], label=f'bias P')
axs[0, 0].set_title(f"Bias at subject {q1+1}")
axs[0, 0].legend()
# Subplot 2
axs[0, 1].plot(P_diff[q2], label='bias P')
axs[0, 1].set_title(f"Bias at subject {q2+1}")
axs[0, 1].legend()
# Subplot 3
axs[1, 0].plot(P_diff[q3], label=f'bias P')
axs[1, 0].set_title(f"Bias at subject {q3+1}")
axs[1, 0].legend()
# Subplot 4
axs[1, 1].plot(P_diff[q4], label='bias P')
axs[1, 1].set_title(f"Bias at subject {q4+1}")
axs[1, 1].legend()
# Adjust layout to prevent overlapping
plt.tight_layout()
# Save the figure
fig.savefig("figures/test.png")

