import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
import nilearn.maskers
import nibabel as nib
import seaborn as sns
import numpy as np
import nilearn
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os

def plot_intensity_1d(G, p, p_hat, filename):
    group_names = list(G.keys())
    n_group = len(group_names)
    n_subject = [len(G[group]) for group in group_names]
    n_voxel = p.shape[1]
    
    fig, axs = plt.subplots(2*n_group, 2, figsize=(10, 10*n_group+1))
    for i, group_name in enumerate(group_names):
        q1 = 0
        q2 = int(np.percentile(np.arange(n_subject[i]), 25))
        q3 = int(np.percentile(np.arange(n_subject[i]), 75))
        q4 = n_subject[i] - 1
        print(f"q1={q1}, q2={q2}, q3={q3}, q4={q4}")

        subject_idx = G[group_name]
        # Subplot 1
        axs[2*i, 0].plot(p_hat[subject_idx,:][q1], label=f'estimated P')
        axs[2*i, 0].plot(p[subject_idx,:][q1], label='actual P')
        axs[2*i, 0].set_title(f"Probability at subject {q1+1} for {group_name} (n_subject={n_subject[i]})")
        axs[2*i, 0].legend()

        # Subplot 2
        axs[2*i, 1].plot(p_hat[subject_idx,:][q2], label='estimated P')
        axs[2*i, 1].plot(p[subject_idx,:][q2], label='actual P')
        axs[2*i, 1].set_title(f"Probability at subject {q2+1} for {group_name} (n_subject={n_subject[i]})")
        axs[2*i, 1].legend()

        # Subplot 3
        axs[2*i+1, 0].plot(p_hat[subject_idx,:][q3], label=f'estimated P')
        axs[2*i+1, 0].plot(p[subject_idx,:][q3], label=f'actual P')
        axs[2*i+1, 0].set_title(f"Probability at subject {q3+1} for {group_name} (n_subject={n_subject[i]})")
        axs[2*i+1, 0].legend()

        # Subplot 4
        axs[2*i+1, 1].plot(p_hat[subject_idx,:][q4], label='estimated P')
        axs[2*i+1, 1].plot(p[subject_idx,:][q4], label='actual P')
        axs[2*i+1, 1].set_title(f"Probability at subject {q4+1} for {group_name} (n_subject={n_subject[i]})")
        axs[2*i+1, 1].legend()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the figure
    fig.savefig(filename)
    
    return 

def plot_intensity_2d(G, p, p_hat, n_voxel, filename):
    group_names = list(G.keys())
    n_group = len(group_names)
    n_subject = [len(G[group]) for group in group_names]    

    fig, axs = plt.subplots(2*n_group, 4, figsize=(46, 20*n_group))
    for i, group_name in enumerate(group_names):
        q1 = 0
        q2 = int(np.percentile(np.arange(n_subject[i]), 25))
        q3 = int(np.percentile(np.arange(n_subject[i]), 75))
        q4 = n_subject[i] - 1

        subject_idx = G[group_name]

        min_value1 = min(p[subject_idx,:][q1,:].min(), p_hat[subject_idx,:][q1,:].min())
        max_value1 = max(p[subject_idx,:][q1,:].max(), p_hat[subject_idx,:][q1,:].max())
        # Subplot 1
        p_1 = p[subject_idx,:][q1].reshape((n_voxel))
        heatmap_1 = axs[2*i, 0].imshow(p_1, cmap='viridis', aspect='equal',vmin=min_value1, vmax=max_value1)
        axs[2*i, 0].set_title(f"Actual probability at subject {q1+1} for group {group_name}", fontsize=30)

        # Subplot 2
        p_hat_1 = p_hat[subject_idx,:][q1].reshape((n_voxel))
        heatmap_2 = axs[2*i, 1].imshow(p_hat_1, cmap='viridis', aspect='equal', vmin=min_value1, vmax=max_value1)
        axs[2*i, 1].set_title(f"Estimated probability at subject {q1+1} for group {group_name}", fontsize=30)
        plt.colorbar(heatmap_2, label='Probability')

        min_value2 = min(p[subject_idx,:][q2,:].min(), p_hat[subject_idx,:][q2,:].min())
        max_value2 = max(p[subject_idx,:][q2,:].max(), p_hat[subject_idx,:][q2,:].max())
        # Subplot 3
        p_2 = p[subject_idx,:][q2].reshape((n_voxel))
        heatmap_3 = axs[2*i, 2].imshow(p_2, cmap='viridis', aspect='equal', vmin=min_value2, vmax=max_value2)
        axs[2*i, 2].set_title(f"Actual probability at subject {q2+1} for group {group_name}", fontsize=30)

        # Subplot 4
        p_hat_2 = p_hat[subject_idx,:][q2].reshape((n_voxel))
        heatmap_4 = axs[2*i, 3].imshow(p_hat_2, cmap='viridis', aspect='equal', vmin=min_value2, vmax=max_value2)
        axs[2*i, 3].set_title(f"Estimated probability at subject {q2+1} for group {group_name}", fontsize=30)
        plt.colorbar(heatmap_4, label='Probability')

        min_value3 = min(p[subject_idx,:][q3,:].min(), p_hat[subject_idx,:][q3,:].min())
        max_value3 = max(p[subject_idx,:][q3,:].max(), p_hat[subject_idx,:][q3,:].max())
        # Subplot 5
        p_3 = p[subject_idx,:][q3].reshape((n_voxel))
        heatmap_5 = axs[2*i+1, 0].imshow(p_3, cmap='viridis', aspect='equal', vmin=min_value3, vmax=max_value3)
        axs[2*i+1, 0].set_title(f"Actual probability at subject {q3+1} for group {group_name}", fontsize=30)

        # Subplot 6
        p_hat_3 = p_hat[subject_idx,:][q3].reshape((n_voxel))
        heatmap_6 = axs[2*i+1, 1].imshow(p_hat_3, cmap='viridis', aspect='equal', vmin=min_value3, vmax=max_value3)
        axs[2*i+1, 1].set_title(f"Estimated probability at subject {q3+1} for group {group_name}", fontsize=30)
        plt.colorbar(heatmap_6, label='Probability')

        min_value4 = min(p[subject_idx,:][q4,:].min(), p_hat[subject_idx,:][q4,:].min())
        max_value4 = max(p[subject_idx,:][q4,:].max(), p_hat[subject_idx,:][q4,:].max())
        # Subplot 7
        p_4 = p[subject_idx,:][q4].reshape((n_voxel))
        heatmap_7 = axs[2*i+1, 2].imshow(p_4, cmap='viridis', aspect='equal', vmin=min_value4, vmax=max_value4)
        axs[2*i+1, 2].set_title(f"Actual probability at subject {q4+1} for group {group_name}", fontsize=30)

        # Subplot 8
        p_hat_4 = p_hat[subject_idx,:][q4].reshape((n_voxel))
        heatmap_8 = axs[2*i+1, 3].imshow(p_hat_4, cmap='viridis', aspect='equal', vmin=min_value4, vmax=max_value4)
        axs[2*i+1, 3].set_title(f"Estimated probability at subject {q4+1} for group {group_name}", fontsize=30)
        plt.colorbar(heatmap_8, label='Probability')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    # Save the figure
    fig.savefig(filename)
    
    return 

def plot_intensity_3d(G, p, p_hat, n_voxel, filename, slice_idx=None):
    group_names = list(G.keys())
    n_group = len(group_names)
    n_subject = [len(G[group]) for group in group_names]    

    fig, axs = plt.subplots(2*n_group, 4, figsize=(46, 20*n_group))
    for i, group_name in enumerate(group_names):
        q1 = 0
        q2 = int(np.percentile(np.arange(n_subject[i]), 25))
        q3 = int(np.percentile(np.arange(n_subject[i]), 75))
        q4 = n_subject[i] - 1

        subject_idx = G[group_name]

        min_value1 = min(p[subject_idx,:][q1,:].min(), p_hat[subject_idx,:][q1,:].min())
        max_value1 = max(p[subject_idx,:][q1,:].max(), p_hat[subject_idx,:][q1,:].max())

        slice_idx = round(0.5*n_voxel[2]) if slice_idx is None else slice_idx
        print("slice_idx", slice_idx)

        # Subplot 1
        p_1 = p[subject_idx,:][q1,:].reshape((n_voxel))
        print(p_1.shape)
        heatmap_1 = axs[2*i, 0].imshow(p_1[:, :, slice_idx], cmap='viridis', aspect='equal')
        axs[2*i, 0].set_title(f"Actual probability at subject {q1+1} for group {group_name}")
        plt.colorbar(heatmap_1, label='Probability')

        # Subplot 2
        p_hat_1 = p_hat[subject_idx,:][q1,:].reshape((n_voxel))
        heatmap_2 = axs[2*i, 1].imshow(p_hat_1[:, :, slice_idx], cmap='viridis', aspect='equal')
        axs[2*i, 1].set_title(f"Estimated probability at subject {q1+1} for group {group_name}")
        plt.colorbar(heatmap_2, label='Probability')

        # Subplot 3
        p_2 = p[subject_idx,:][q2,:].reshape((n_voxel))
        heatmap_3 = axs[2*i, 2].imshow(p_2[:, :, slice_idx], cmap='viridis', aspect='equal')
        axs[2*i, 2].set_title(f"Actual probability at subject {q2+1} for group {group_name}")
        plt.colorbar(heatmap_3, label='Probability')

        # Subplot 4
        p_hat_2 = p_hat[subject_idx,:][q2,:].reshape((n_voxel))
        heatmap_4 = axs[2*i, 3].imshow(p_hat_2[:, :, slice_idx], cmap='viridis', aspect='equal')
        axs[2*i, 3].set_title(f"Estimated probability at subject {q2+1} for group {group_name}")
        plt.colorbar(heatmap_4, label='Probability')


        # Subplot 5
        p_3 = p[subject_idx,:][q3,:].reshape((n_voxel))
        heatmap_5 = axs[2*i+1, 0].imshow(p_3[:, :, slice_idx], cmap='viridis', aspect='equal')
        axs[2*i+1, 0].set_title(f"Actual probability at subject {q3+1} for group {group_name}")
        plt.colorbar(heatmap_5, label='Probability')

        # Subplot 6
        p_hat_3 = p_hat[subject_idx,:][q3,:].reshape((n_voxel))
        heatmap_6 = axs[2*i+1, 1].imshow(p_hat_3[:, :, slice_idx], cmap='viridis', aspect='equal')
        axs[2*i+1, 1].set_title(f"Estimated probability at subject {q3+1} for group {group_name}")
        plt.colorbar(heatmap_6, label='Probability')

        # Subplot 7
        p_4 = p[subject_idx,:][q4,:].reshape((n_voxel))
        heatmap_7 = axs[2*i+1, 2].imshow(p_4[:, :, slice_idx], cmap='viridis', aspect='equal')
        axs[2*i+1, 2].set_title(f"Actual probability at subject {q4+1} for group {group_name}")
        plt.colorbar(heatmap_7, label='Probability')

        # Subplot 8
        p_hat_4 = p_hat[subject_idx,:][q4,:].reshape((n_voxel))
        heatmap_8 = axs[2*i+1, 3].imshow(p_hat_4[:, :, slice_idx], cmap='viridis', aspect='equal')
        axs[2*i+1, 3].set_title(f"Estimated probability at subject {q4+1} for group {group_name}")
        plt.colorbar(heatmap_8, label='Probability')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    # Save the figure
    fig.savefig(filename)
    
    return 

def plot_brain(p, brain_mask, slice_idx=None, threshold=5e-4, vmin=0, vmax=None, output_filename="test.png"):
    print("threshold", threshold)
    brain_mask_data = brain_mask.get_fdata()
    mask_indices = np.where(brain_mask_data > 0)
    if len(p) != len(mask_indices[0]):
        raise ValueError("The number of voxels in the probability map is not equal to the number of voxels in the brain mask")
    # nan for p-values outside of brain mask
    nifti_data = np.zeros(brain_mask_data.shape, dtype=np.float32)
    # Assign p-vals/z-statistics to the masked voxels
    nifti_data[mask_indices] = p.ravel()
    # # Only display the values below the threshold 0.05
    # Create a new NIfTI image
    nifti_image = nib.Nifti1Image(nifti_data, affine=brain_mask.affine, header=brain_mask.header)
    plot_stat_map(
        nifti_image, 
        cut_coords=[0,6,12,18,24,30,36],
        display_mode='z', 
        draw_cross=False, 
        cmap='inferno',
        threshold=threshold,
        colorbar=True,
        vmin=vmin,
        vmax=vmax,
        output_file=output_filename
    )
    print(output_filename)
    return

def save_nifti(p, brain_mask, output_filename):
    brain_mask_data = brain_mask.get_fdata()
    mask_indices = np.where(brain_mask_data > 0)
    if len(p) != len(mask_indices[0]):
        raise ValueError("The number of voxels in the probability map is not equal to the number of voxels in the brain mask")
    nifti_data = np.zeros(brain_mask_data.shape, dtype=np.float32)
    # Assign p-vals/z-statistics to the masked voxels
    nifti_data[mask_indices] = p.ravel()
    # NaN for p-values outside of brain mask
    nifti_data[nifti_data == 0] = np.nan
    # Create a new NIfTI image
    nifti_image = nib.Nifti1Image(nifti_data, affine=brain_mask.affine, header=brain_mask.header)
    nib.save(nifti_image, output_filename)
    print(f"Saved NIfTI file to {output_filename}")
    return