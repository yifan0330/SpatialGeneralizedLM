import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import OneHotEncoder
import nibabel as nib
import nilearn
import matplotlib.pyplot as plt
import scipy.stats
import hashlib
import os
import time

class simulated_data(object):
    def __init__(self, space_dim, n_group, n_subject=2000, n_voxel=1000, brain_mask=None, 
                 group_names=None, homogeneous_intensity=True, lesion_per_subject=10):
        self.space_dim = space_dim
        self.group_names = group_names if group_names is not None else [f"group_{i}" for i in range(n_group)]
        self.n_group = n_group
        self.n_subject = n_subject
        self.n_voxel = brain_mask._dataobj.shape if space_dim == "brain" else n_voxel
        self.brain_mask = brain_mask
        self.homogeneous_intensity = homogeneous_intensity
        if len(lesion_per_subject) != n_group:
            raise ValueError(f"Length of lesion per subject: {len(lesion_per_subject)} doesn't equal to number of groups: {n_group}")
        # create underlying intensity funtion 
        self.background_intensity_func = self.create_background_intensity_func(lesion_per_subject, brain_mask)
        self.covariate_intensity_func = self.create_covariate_intensity_func(lesion_per_subject, brain_mask)

    def create_background_intensity_func(self, lesion_per_subject, brain_mask, cov_scale=100):
        background_intensity_func = dict()
        if self.homogeneous_intensity:
            if self.space_dim in [1,2,3]:
                for i in range(self.n_group):
                    group_name = self.group_names[i]
                    background_intensity_func[group_name] = lesion_per_subject[i]/np.prod(self.n_voxel)*np.ones(self.n_voxel)
            elif self.space_dim == "brain":
                for i in range(self.n_group):
                    group_name = self.group_names[i]
                    background_intensity_func[group_name] = 2*lesion_per_subject[i]/np.prod(self.n_voxel)*np.ones(self.n_voxel)
        else:
            if self.space_dim == 1:
                x = np.linspace(0,self.n_voxel[0]-1, self.n_voxel[0])
                for i in range(self.n_group):
                    group_name = self.group_names[i]
                    background_intensity_func[group_name] = lesion_per_subject[i]*scipy.stats.norm.pdf(x, loc=np.mean(x), scale=cov_scale)
            elif self.space_dim == 2:
                x = np.linspace(0,self.n_voxel[0]-1, self.n_voxel[0])
                y = np.linspace(0,self.n_voxel[1]-1, self.n_voxel[1])
                X, Y = np.meshgrid(x, y, indexing='ij')
                coordinates = np.stack([X.ravel(), Y.ravel()], axis=-1)
                bump_mean = [round(np.mean(x)), round(np.mean(y))]
                bump_cov = cov_scale*np.eye(self.space_dim)
                # background intensity function
                for i in range(self.n_group):
                    group_name = self.group_names[i]
                    background_intensity_func[group_name] = lesion_per_subject[i]*(scipy.stats.multivariate_normal.cdf(coordinates+[0.5,0.5], mean=bump_mean, cov=bump_cov) \
                        - scipy.stats.multivariate_normal.cdf(coordinates+[0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                        - scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,0.5], mean=bump_mean, cov=bump_cov) \
                        + scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,-0.5], mean=bump_mean, cov=bump_cov))
                    background_intensity_func[group_name] += 0.01

            elif self.space_dim == 3:
                filename = f"probability_function/{self.space_dim}D_{self.n_group}_group_bump_background_intensity_func.npz"
                if os.path.exists(filename):
                    background_intensity_func = np.load(filename)
                    background_intensity_func = {key: background_intensity_func[key] for key in background_intensity_func.files}
                else:
                    x = np.linspace(0,self.n_voxel[0]-1, self.n_voxel[0])
                    y = np.linspace(0,self.n_voxel[1]-1, self.n_voxel[1])
                    z = np.linspace(0,self.n_voxel[2]-1, self.n_voxel[2])
                    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                    coordinates = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
                    bump_mean = [round(np.mean(x)), round(np.mean(y)), round(np.mean(z))]
                    bump_cov = cov_scale*np.eye(self.space_dim)
                    # background intensity function
                    for i in range(self.n_group):
                        group_name = self.group_names[i]
                        background_intensity_func[group_name] = lesion_per_subject[i]*(scipy.stats.multivariate_normal.cdf(coordinates+[0.5,0.5,0.5], mean=bump_mean, cov=bump_cov) \
                            - scipy.stats.multivariate_normal.cdf(coordinates+[0.5,0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                            - scipy.stats.multivariate_normal.cdf(coordinates+[0.5,-0.5,0.5], mean=bump_mean, cov=bump_cov) \
                            - scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,0.5,0.5], mean=bump_mean, cov=bump_cov) \
                            - scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,-0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                            + scipy.stats.multivariate_normal.cdf(coordinates+[0.5,-0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                            + scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                            + scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,-0.5,0.5], mean=bump_mean, cov=bump_cov))
                        background_intensity_func[group_name] += 0.01
                    np.savez(filename, **background_intensity_func)
            elif self.space_dim == "brain":
                raise NotImplementedError("Brain template not implemented")

        return background_intensity_func
    
    def create_covariate_intensity_func(self, lesion_per_subject, brain_mask, cov_scale=100):
        covariate_intensity_func = dict()
        if self.homogeneous_intensity:
            if self.space_dim == 1:
                start_index, end_index = round(0.25*self.n_voxel[0]), round(0.75*self.n_voxel[0])
                for i in range(self.n_group):
                    group_name = self.group_names[i]
                    covariate_intensity_func[group_name] = np.zeros(self.n_voxel)
                    covariate_intensity_func[group_name][start_index:end_index] = 0.5*2**self.space_dim*lesion_per_subject[i]/np.prod(self.n_voxel)
            elif self.space_dim == 2:
                start_index_0, end_index_0 = round(0.25*self.n_voxel[0]), round(0.75*self.n_voxel[0])
                start_index_1, end_index_1 = round(0.4*self.n_voxel[1]), round(0.6*self.n_voxel[1])
                for i in range(self.n_group):
                    group_name = self.group_names[i]
                    covariate_intensity_func[group_name] = np.zeros(self.n_voxel)
                    covariate_intensity_func[group_name][start_index_0:end_index_0, start_index_1:end_index_1] = 0.5*2**self.space_dim*lesion_per_subject[i]/np.prod(self.n_voxel)
            elif self.space_dim == 3:
                start_index_0, end_index_0 = round(0.25*self.n_voxel[0]), round(0.75*self.n_voxel[0])
                start_index_1, end_index_1 = round(0.4*self.n_voxel[1]), round(0.6*self.n_voxel[1])
                start_index_2, end_index_2 = round(0.25*self.n_voxel[2]), round(0.75*self.n_voxel[2])
                for i in range(self.n_group):
                    group_name = self.group_names[i]
                    covariate_intensity_func[group_name] = np.zeros(self.n_voxel)
                    covariate_intensity_func[group_name][start_index_0:end_index_0, start_index_1:end_index_1, start_index_2:end_index_2] = 0.5*2**self.space_dim*lesion_per_subject[i]/np.prod(self.n_voxel)
            elif self.space_dim == "brain":
                start_index_0, end_index_0 = round(0.25*self.n_voxel[0]), round(0.75*self.n_voxel[0])
                start_index_1, end_index_1 = round(0.4*self.n_voxel[1]), round(0.6*self.n_voxel[1])
                start_index_2, end_index_2 = round(0.25*self.n_voxel[2]), round(0.75*self.n_voxel[2])
                covariate_intensity_func = np.zeros((self.n_voxel))
                covariate_intensity_func[start_index_0:end_index_0, start_index_1:end_index_1, start_index_2:end_index_2] = 8*lesion_per_subject/np.prod(self.n_voxel)
        else:
            if self.space_dim == 1:
                x = np.linspace(0,self.n_voxel[0]-1, self.n_voxel[0])
                for i in range(self.n_group):
                    group_name = self.group_names[i]
                    covariate_intensity_func[group_name] = 0.5*lesion_per_subject[i]*scipy.stats.norm.pdf(x, loc=np.mean(x), scale=0.5*cov_scale)
            elif self.space_dim == 2:
                x = np.linspace(0,self.n_voxel[0]-1, self.n_voxel[0])
                y = np.linspace(0,self.n_voxel[1]-1, self.n_voxel[1])
                X, Y = np.meshgrid(x, y, indexing='ij')
                coordinates = np.stack([X.ravel(), Y.ravel()], axis=-1)
                bump_mean = [round(np.mean(x)), round(np.mean(y))]
                bump_cov = 0.5*cov_scale*np.eye(self.space_dim)
                for i in range(self.n_group):
                    group_name = self.group_names[i]
                    covariate_intensity_func[group_name] = 0.25*lesion_per_subject[i]*(scipy.stats.multivariate_normal.cdf(coordinates+[0.5,0.5], mean=bump_mean, cov=bump_cov) \
                        - scipy.stats.multivariate_normal.cdf(coordinates+[0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                        - scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,0.5], mean=bump_mean, cov=bump_cov) \
                        + scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,-0.5], mean=bump_mean, cov=bump_cov))
            elif self.space_dim == 3:
                filename = f"probability_function/{self.space_dim}D_{self.n_group}_group_bump_covariate_intensity_func.npy"
                if os.path.exists(filename):
                    covariate_intensity_func = np.load(filename)
                    covariate_intensity_func = {key: background_intensity_func[key] for key in covariate_intensity_func.files}
                else:
                    x = np.linspace(0,self.n_voxel[0]-1, self.n_voxel[0])
                    y = np.linspace(0,self.n_voxel[1]-1, self.n_voxel[1])
                    z = np.linspace(0,self.n_voxel[2]-1, self.n_voxel[2])
                    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                    coordinates = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
                    bump_mean = [round(np.mean(x)), round(np.mean(y)), round(np.mean(z))]
                    bump_cov = 0.5*cov_scale*np.eye(self.space_dim)
                    for i in range(self.n_group):
                        group_name = self.group_names[i]
                        covariate_intensity_func[group_name] = 0.25*lesion_per_subject[i]*(scipy.stats.multivariate_normal.cdf(coordinates+[0.5,0.5,0.5], mean=bump_mean, cov=bump_cov) \
                            - scipy.stats.multivariate_normal.cdf(coordinates+[0.5,0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                            - scipy.stats.multivariate_normal.cdf(coordinates+[0.5,-0.5,0.5], mean=bump_mean, cov=bump_cov) \
                            - scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,0.5,0.5], mean=bump_mean, cov=bump_cov) \
                            - scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,-0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                            + scipy.stats.multivariate_normal.cdf(coordinates+[0.5,-0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                            + scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,0.5,-0.5], mean=bump_mean, cov=bump_cov) \
                            + scipy.stats.multivariate_normal.cdf(coordinates+[-0.5,-0.5,0.5], mean=bump_mean, cov=bump_cov))
                    np.savez(filename, **covariate_intensity_func)
            elif self.space_dim == "brain":
                raise NotImplementedError("Brain template not implemented")
        return covariate_intensity_func

    def scale_constant(self, space_dim, n_neighbor, lesion_size_max):
        space_dim = 3 if self.space_dim == "brain" else self.space_dim
        sequence = [1 / lesion_size_max * (i+1)/3**space_dim for i in range(lesion_size_max)]
        sequence_sum = sum(sequence)
        scale_constant = sequence_sum * n_neighbor

        return scale_constant
    
    def generate_data(self, lesion_size_range):
        Y, total_intensity_func = list(), list()
        # covariate effect: subject index 
        subject_idx = [np.arange(start=0, stop=1, step=1/self.n_subject[i], dtype=np.float64) for i in range(self.n_group)]
        subject_idx = np.concatenate(subject_idx, axis=0).reshape(-1,1)
        # covariate effect: group
        subject_group = np.concatenate([[i]*self.n_subject[i] for i in range(self.n_group)], axis=0)
        # convert to one-hot encoding
        group_one_hot_encoder = np.stack([(subject_group == i).astype(int) for i in range(self.n_group)], axis=1)
        Z = np.concatenate([subject_idx, group_one_hot_encoder], axis=1) # shape: (n_subject, n_covariate)
        # subject indices in different groups based on categorical covariate
        group_subjects = {f"group_{int(key)}": list(np.where(subject_group == key)[0]) for key in range(self.n_group)}
        # other spatial effect shared by subjects
        seed = 0
        for k in range(self.n_group):
            group_name = self.group_names[k]
            subject_indices = group_subjects[group_name]
            for s in subject_indices:
                np.random.seed(seed)
                # combine background intensity and spatially varying covariate intensity
                sum_intensity_func_s = self.background_intensity_func[group_name] + Z[s,0]*self.covariate_intensity_func[group_name]
                sum_intensity_func_s = np.clip(sum_intensity_func_s, a_min=0, a_max=1)
                # generate brain lesion for each subject
                sum_intensity_func_s = sum_intensity_func_s.reshape((self.n_voxel))
                lesion_centre = np.random.binomial(n=1, p=sum_intensity_func_s, size=self.n_voxel)
                n_lesion_centre = np.sum(lesion_centre)
                if self.space_dim == 1:
                    lesion_centriod = np.where(lesion_centre==1)[0]
                elif self.space_dim == 2:
                    lesion_centriod_x, lesion_centriod_y = np.where(lesion_centre==1)
                elif self.space_dim in [3, "brain"]:
                    lesion_index_x, lesion_index_y, lesion_index_z = np.where(lesion_centre==1)
                Y_s = np.zeros(self.n_voxel).astype(np.int32)
                # Randomly generate lesion at neighbouring voxels based on lesion size
                lesion_size = np.random.randint(lesion_size_range[0],lesion_size_range[1]+1,size=n_lesion_centre)
                if self.space_dim == 1:
                    offset = [-1, 0, 1]
                    n_neighbor = len(offset)
                    for i in range(n_lesion_centre):
                        sampled_offsets = np.random.choice(n_neighbor, size=lesion_size[i], replace=False)
                        sampled_neighbors = [lesion_centriod[i] + offset[j] for j in sampled_offsets]
                        for neighbor in sampled_neighbors:
                            if 0 <= neighbor < self.n_voxel:
                                Y_s[neighbor] += 1
                elif self.space_dim == 2:
                    # Define all possible neighbor offsets (3x3 square, excluding center)
                    offsets = [(dx, dy)
                                for dx in [-1, 0, 1]
                                for dy in [-1, 0, 1]]
                    n_neighbor = len(offsets)
                    for i in range(n_lesion_centre):
                        sampled_offsets = np.random.choice(n_neighbor, size=lesion_size[i], replace=False)
                        sampled_neighbors = [tuple(np.add([lesion_centriod_x[i], lesion_centriod_y[i]], offsets[j])) for j in sampled_offsets]
                        for neighbor in sampled_neighbors:
                            if 0 <= neighbor[0] < self.n_voxel[0] and 0 <= neighbor[1] < self.n_voxel[1]:
                                Y_s[neighbor] += 1
                elif self.space_dim in [3, "brain"]:
                    # Define all possible neighbor offsets (3x3x3 cube, excluding center)
                    offsets = [(dx, dy, dz)
                                for dx in [-1, 0, 1]
                                for dy in [-1, 0, 1]
                                for dz in [-1, 0, 1]]
                    n_neighbor = len(offsets)
                    for i in range(n_lesion_centre):
                        sampled_offsets = np.random.choice(n_neighbor, size=lesion_size[i], replace=False)
                        sampled_neighbors = [tuple(np.add([lesion_index_x[i], lesion_index_y[i], lesion_index_z[i]], offsets[j])) for j in sampled_offsets]
                        for neighbor in sampled_neighbors:
                            if 0 <= neighbor[0] < self.n_voxel[0] and 0 <= neighbor[1] < self.n_voxel[1] and 0 <= neighbor[2] < self.n_voxel[2]:
                                Y_s[neighbor] += 1
                Y_s = np.clip(Y_s, a_min=0, a_max=1)
                # Scale intensity function by corresponding constant
                C = self.scale_constant(self.space_dim, n_neighbor, lesion_size_range[1])
                total_intensity_func_s = C*sum_intensity_func_s
                # Reshape data
                Y_s = Y_s.reshape((1, -1)) if self.space_dim != "brain" else Y_s[None, ...]
                total_intensity_func_s = total_intensity_func_s.reshape((1, -1)) if self.space_dim != "brain" else total_intensity_func_s[None, ...]
                # Store data
                Y.append(Y_s)
                total_intensity_func.append(total_intensity_func_s)
                seed += 1
                # # 1D Visualisation
                # plt.figure(figsize=(100, 2))
                # # plt.step(range(self.n_voxel), Y_s, where="mid")
                # plt.step(range(self.n_voxel), 0.5*lesion_size_range[1]*sum_intensity_func_s, where="mid")
                # plt.xlabel("Voxel location")
                # plt.ylabel("Brain lesion")
                # plt.title("1D simulation of brain lesion on 1000 voxels")
                # plt.savefig("test.png")
                # # 2D Visualisation
                # plt.figure(figsize=(8, 8))
                # plt.imshow(total_intensity_func_s, cmap='viridis', aspect='equal')
                # # plt.imshow(Y_s, cmap='viridis', aspect='equal')
                # plt.colorbar(label='Intensity')  # Add a color bar with a label
                # plt.xlabel('X-axis')
                # plt.ylabel('Y-axis')
                # plt.savefig("test.png")
                # 3D Visualisation
                # plt.figure(figsize=(8, 8))
                # plt.imshow(total_intensity_func_s[:,:,10], cmap='viridis', aspect='equal')
                # # plt.imshow(total_intensity_func_s[:,:,10].reshape((self.n_voxel[0], self.n_voxel[1])), cmap='viridis', aspect='equal')
                # # plt.imshow(Y_s.reshape(self.n_voxel)[:,:,10], cmap='viridis', aspect='equal')
                # plt.colorbar(label='Intensity')  # Add a color bar with a label
                # plt.xlabel('X-axis')
                # plt.ylabel('Y-axis')
                # plt.savefig("test.png")
        Y = np.concatenate(Y, axis=0)
        total_intensity_func = np.concatenate(total_intensity_func, axis=0)
        # if self.space_dim == "brain": 
        #     Y = Y[:,self.brain_mask._dataobj>0]
        #     total_intensity_func = total_intensity_func[:,self.brain_mask._dataobj>0]
        return group_subjects, total_intensity_func, Y, Z

class GRF_simulated_data(object):
    def __init__(self, GRF_data_dir, n_group, n_subject, group_names=None):
        self.group_names = group_names
        self.n_group = n_group
        self.GRF_data_dir = GRF_data_dir
        self.n_subject = n_subject
        # check if data directory exists
        if not os.path.exists(self.GRF_data_dir):
            raise ValueError(f"GRF data directory {self.GRF_data_dir} doesn't exist")
    
    def simulate_grf_gaussian(self, shape, var=1, scale=1.5, random_seed=42):
        """
        Simulate a 3D Gaussian random field with target variance using
        Gaussian smoothing of white noise as an approximation to RMgauss.

        Parameters
        ----------
        shape : tuple[int, int, int]
            Volume shape (e.g., (91, 109, 91))
        var : float
            Desired marginal variance of the field
        scale : float
            Correlation scale (voxel units). Interpreted as sigma for gaussian_filter.
            (In RandomFields, covariance ~ exp(-(h/scale)^2). For a quick approximation,
            using sigma ~ scale works reasonably in practice.)
        rng : np.random.Generator
            Random generator

        Returns
        -------
        grf : np.ndarray
            3D array with approximately N(0, var) marginal distribution.
        """
        rng = np.random.default_rng(random_seed)
        wn = rng.standard_normal(size=shape).astype(np.float32)
        # Smooth white noise; mode='reflect' to avoid edge artifacts
        sm = scipy.ndimage.gaussian_filter(wn, sigma=scale, mode="reflect")
        # Rescale to desired variance
        sm_mean = sm.mean()
        sm_std = sm.std(ddof=0)
        if sm_std == 0:
            raise RuntimeError("Gaussian smoothing produced zero variance; check parameters.")
        grf = (sm - sm_mean) * (np.sqrt(var) / sm_std)
        return grf.astype(np.float32)

    def combine_seeds_int(self, seed1: int, seed2: int) -> int:
        # Convert both integers to bytes (8 bytes each, little-endian)
        s = seed1.to_bytes(8, "little", signed=False) + seed2.to_bytes(8, "little", signed=False)
        digest = hashlib.sha256(s).digest()
        # Take the first 4 bytes → 32-bit integer seed
        return int.from_bytes(digest[:4], "little")

    def process_data(self, mask_path, low=45.12, high=80.65, 
                     target_voxel_size=(2.0,2.0,2.0), random_seed=42):
        # load brain mask 
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata().astype(bool)
        n_voxel = np.sum(mask_data)
        mask_shape = mask_data.shape
        
        coef_age_img = nib.load(self.GRF_data_dir+"coef_age_nvars_1_method_2.nii.gz")
        coef_age_ukbb = coef_age_img.get_fdata().astype(np.float32)
        coef_int_img = nib.load(self.GRF_data_dir+"coef_Intercept_nvars_1_method_2.nii.gz")
        coef_intercept_ukbb = coef_int_img.get_fdata().astype(np.float32)
        empir_prob_img = nib.load(self.GRF_data_dir+"empir_prob_mask.nii.gz")
        empir_prob = empir_prob_img.get_fdata().astype(np.float32)

        # age
        rng_age = np.random.default_rng(random_seed)
        outcome = dict()
        for i in range(self.n_group):
            n = self.n_subject[i]
            group_name = self.group_names[i]
            ages = rng_age.uniform(low=low, high=high, size=n)
            ages = np.round(np.sort(ages), 2)
            # create empty dict and arrays
            outcome[group_name] = dict()
            Z_i = ages.copy().reshape((n, -1))
            Y_i = np.empty((n, n_voxel), dtype=np.float32)
            for i in range(n):
                grf_seed = self.combine_seeds_int(random_seed, i)
                # simulated GRF
                grf = self.simulate_grf_gaussian(mask_shape, var=1, scale=1.5, random_seed=grf_seed)
                # Precompute xbeta (intercept only in your current R)
                xbeta = coef_intercept_ukbb
                # Compute z = pnorm(grf + xbeta), deterministic threshold at 0.5
                z = scipy.stats.norm.cdf(grf + xbeta)
                # Binary lesion map
                y_bin = np.where(z >= 0.5, 1.0, 0.0).astype(np.float32)
                # Apply masks
                y_bin[mask_data == 0] = 0.0
                y_bin[empir_prob == 0] = 0.0
                y_bin = y_bin[mask_data]
                Y_i[i, :] = y_bin
            outcome[group_name]['Z'] = Z_i
            outcome[group_name]['Y'] = Y_i
        return outcome

class SpatialHomo_simulated_data(object):
    def __init__(self, HOMO_data_dir, n_group, n_subject, group_names=None):
        self.group_names = group_names if group_names is not None else [f"group_{i}" for i in range(n_group)]
        self.n_group = n_group
        self.HOMO_data_dir = HOMO_data_dir
        self.n_subject = n_subject
        # check if data directory exists
        if not os.path.exists(self.HOMO_data_dir):
            raise ValueError(f"HOMO data directory {self.HOMO_data_dir} doesn't exist")

    def process_data(self, mask_path, n_subject=1000, low=45.12, high=80.65, 
                     target_voxel_size=(2.0,2.0,2.0), random_seed=42):
        # load brain mask 
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata().astype(bool)
        n_voxel = np.sum(mask_data)
        mask_shape = mask_data.shape
        
        coef_age_img = nib.load(self.HOMO_data_dir+"coef_age_nvars_1_method_2.nii.gz")
        coef_age_ukbb = coef_age_img.get_fdata().astype(np.float32)
        coef_int_img = nib.load(self.HOMO_data_dir+"coef_Intercept_nvars_1_method_2.nii.gz")
        coef_intercept_ukbb = coef_int_img.get_fdata().astype(np.float32)
        empir_prob_img = nib.load(self.HOMO_data_dir+"empir_prob_mask.nii.gz")
        empir_prob = empir_prob_img.get_fdata().astype(np.float32)

        # age
        print("random seed", random_seed)
        rng = np.random.default_rng(random_seed)
        ages = rng.uniform(low=low, high=high, size=n_subject)
        ages = np.round(np.sort(ages), 2)

        Z = ages.copy().reshape((n_subject, -1))
        Y = np.empty((n_subject, n_voxel), dtype=np.float32)
        for i in range(n_subject):
            y_bin = rng.choice([0, 1], size=mask_shape, p=[0.99, 0.01])
            # Apply masks
            y_bin[mask_data == 0] = 0.0
            y_bin[empir_prob == 0] = 0.0
            y_bin = y_bin[mask_data]
            Y[i, :] = y_bin

        return Z, Y

class SubjectHomo_simulated_data(object):
    def __init__(self, HOMO_data_dir, n_group, n_subject, group_names=None):
        self.group_names = group_names if group_names is not None else [f"group_{i}" for i in range(n_group)]
        self.n_group = n_group
        self.HOMO_data_dir = HOMO_data_dir
        self.n_subject = n_subject
        # check if data directory exists
        if not os.path.exists(self.HOMO_data_dir):
            raise ValueError(f"HOMO data directory {self.HOMO_data_dir} doesn't exist")

    def process_data(self, mask_path, n_subject=1000, low=45.12, high=80.65, 
                     target_voxel_size=(2.0,2.0,2.0), random_seed=42):
        # load brain mask 
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata().astype(bool)
        n_voxel = np.sum(mask_data)
        mask_shape = mask_data.shape
        
        # coef_age_img = nib.load(self.HOMO_data_dir+"coef_age_nvars_1_method_2.nii.gz")
        # coef_age_ukbb = coef_age_img.get_fdata().astype(np.float32)
        # coef_int_img = nib.load(self.HOMO_data_dir+"coef_Intercept_nvars_1_method_2.nii.gz")
        # coef_intercept_ukbb = coef_int_img.get_fdata().astype(np.float32)
        empir_prob_img = nib.load(self.HOMO_data_dir+"empir_prob_mask.nii.gz")
        empir_prob = empir_prob_img.get_fdata().astype(np.float32)

        # age
        print("random seed", random_seed)
        rng = np.random.default_rng(random_seed)
        ages = rng.uniform(low=low, high=high, size=n_subject)
        ages = np.round(np.sort(ages), 2)

        Z = ages.copy().reshape((n_subject, -1))
        Y = np.empty((n_subject, n_voxel), dtype=np.float32)
        for i in range(n_subject):
            y_bin = rng.choice([0, 1], p=[0.5, 0.5])
            y_bin = np.full(n_voxel, y_bin)
            Y[i, :] = y_bin

        return Z, Y




class Biobank_data(object):
    def __init__(self, data_dir, subject_data_dir):
        self.data_dir = data_dir
        self.subject_data_dir = subject_data_dir
        # check if data directory exists
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Brain lesion data directory {self.data_dir} doesn't exist")
        if not os.path.exists(self.subject_data_dir):
            raise ValueError(f"Subject data directory {self.subject_data_dir} doesn't exist")
        # check if data has been pre-processed
        processed_data_dir = os.path.dirname(os.getcwd()) + "/UKB_data/"
    
    def process_data(self, mask_path, target_voxel_size=(2.0,2.0,2.0)):
        # load brain mask 
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata().astype(bool)
        print(mask_data.shape, np.sum(mask_data))
        # csf_img_path = os.path.join(self.subject_data_dir, f"avg152T1_csf.hdr")
        # csf_img = nib.load(csf_img_path)
        # csf_data = np.squeeze(csf_img.get_fdata(), axis=-1)
        # csf_prob_bool = csf_data <= 0.5
        # mask_data = np.logical_and(mask_data, csf_prob_bool)
        n_voxel = np.sum(mask_data)
        # load subject data
        subject_data = pd.read_csv(self.subject_data_dir + "GLM_CVR.dat", delimiter="\t")
        subject_id_bridge = pd.read_csv(self.subject_data_dir + "bridge_8107_34077.tsv", delimiter="\t")
        # merge subject eid_8107 with eid_34077
        subject_data_merge = pd.merge(subject_data, subject_id_bridge, left_on="eid_8107", right_on="eid")
        Z = subject_data_merge.loc[:, ["eid34077", "sexF", "age", "headsize", "CVR6"]]
        # Z_indexed = Z.set_index("eid_34077")
        n_subject = Z.shape[0]
        # load brain lesion data
        print("n_subject: ", n_subject, "n_voxel: ", n_voxel)
        Y = np.memmap("data/UKB/Y_lesion.dat", dtype="int8", mode="w+", shape=(n_subject, n_voxel))
        subject_eid = Z["eid34077"].values
        missing_subject_id, missing_subject_eid = list(), list()
        id = 0
        for eid in subject_eid:
            lesion_data_path = os.path.join(self.data_dir, f"2{eid}/T2_FLAIR/lesions/final_mask_to_MNI.nii.gz")
            if os.path.exists(lesion_data_path):
                lesion_img = nib.load(lesion_data_path)
                lesion_data = lesion_img.get_fdata()
                affine = lesion_img.affine
                orig_voxel_size = lesion_img.header.get_zooms()  # Example: (1.0, 1.0, 1.0)
                # Compute resampling factor
                zoom_factors = np.array(orig_voxel_size) / np.array(target_voxel_size)
                # Resample using cubic interpolation
                resampled_lesion_data = scipy.ndimage.zoom(lesion_data, zoom_factors, order=3).astype(np.int8)
                Y[id, :] = resampled_lesion_data[mask_data]
                Y.flush()
            else:
                print(f"Subject {id, eid} doesn't have lesion data")
                missing_subject_id.append(id)
                missing_subject_eid.append(eid)
            id += 1
        Z = Z[~Z["eid34077"].isin(missing_subject_eid)]
        Y = np.delete(Y, missing_subject_id, axis=0)
        self.Z, self.Y = Z, Y

        return Z, Y

