import numpy as np
from scipy.optimize import curve_fit
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import vec_val_vect
from dipy.core.geometry import (euler_matrix,
                                vec2vec_rotmat, decompose_matrix,
                                rodrigues_axis_rotation)
from dipy.reconst.base import ReconstFit, ReconstModel
from dipy.reconst.dti import (TensorModel, TensorFit)
from dipy.reconst.ivim import IvimModel
from tqdm import tqdm
#from tqdm.asyncio import tqdm
from concurrent.futures import ThreadPoolExecutor
import asyncio
import nest_asyncio
nest_asyncio.apply()

def _ivim_tensor_equation(beta, b, bvecs, Q_star, Q):
    """ 
    $\frac{S}{S_0} = \beta e^{-b \theta^t Q^{*}, \theta} + (1- \beta) e^{-b \theta^t Q \theta}$
    """
    return beta * np.exp(-b * np.diag(bvecs @ Q_star @ bvecs.T)) + (1 - beta) * np.exp(-b * np.diag(bvecs @ Q @ bvecs.T))


def _reconstruct_tensor(eval0, eval1, eval2, eul0, eul1, eul2):
    """
    Reconstruct 12 tensor params from eigenvalues and Euler rotations.
    """
    R = euler_matrix(eul0, eul1, eul2)
    evecs = R[:3, :3]
    evals = np.array([eval0, eval1, eval2])
    return evecs, evals


def _reconstruct_tensors(params):
    """
    Reconstruct two quadratic forms of tensors based on 
    12 params (two sets of eigenvalues and Euler angles).
    """
    Q = vec_val_vect(*_reconstruct_tensor(
        params[0], params[1], params[2],
        params[3], params[4], params[5]))

    Q_star = vec_val_vect(*_reconstruct_tensor(
        params[6], params[7], params[8],
        params[9], params[10], params[11]))

    return Q, Q_star


def calc_euler(evecs):
    """
    Calculate the Euler angles that rotate from the canonical coordinate frame
    to a coordinate frame defined by a set of eigenvectors.

    Parameters
    ----------
    evecs : 3-by-3 array

    """
    rot0 = np.eye(4)
    # What is the rotation from the first eigenvector to eye(3)?
    rot0[:3, :3] = vec2vec_rotmat(evecs[0], np.eye(3)[0])
    # Decompose (we only need the angles)
    scale, shear, angles0, translate, perspective = decompose_matrix(rot0)
    # Convert angles to Euler matrix:
    em = euler_matrix(*angles0)
    # Now, we need another rotation to bring the second eigenvector to the right
    # direction
    ang1 = np.arccos(
        np.dot(evecs[1], em[1, :3])
        / (np.linalg.norm(evecs[1]) * np.linalg.norm(em[1, :3])))
    rar = np.eye(4)
    # The rar is a matrix that rotates for a given angle around a given
    # vector:
    rar[:3, :3] = rodrigues_axis_rotation(evecs[0], np.rad2deg(ang1))
    # We combine these two rotations and decompose the combined matrix to give
    # us three Euler angles, which will be our parameters
    scale, shear, angles, translate, perspective = decompose_matrix(em @ rar)
    return angles


class IvimTensorModel(ReconstModel):
    def __init__(self, gtab, split_b_D=200.0, n_threads=1):
        """
        Model to reconstruct an IVIM tensor

        Parameters
        ----------
        gtab : GradientTable class instance

        split_b_D : float
            The value of b that separates perfusion from diffusion
        """
        ReconstModel.__init__(self, gtab)
        self.split_b_D = split_b_D
        # Use two separate tensors for initial estimation:
        self.diffusion_idx = np.hstack([np.where(gtab.bvals > self.split_b_D),
                                        np.where(gtab.b0s_mask)]).squeeze()
        
        # The first tensor represents diffusion
        self.diffusion_gtab = gradient_table(
            self.gtab.bvals[self.diffusion_idx],
            self.gtab.bvecs[self.diffusion_idx])

        self.diffusion_model = TensorModel(self.diffusion_gtab)

        # The second tensor represents perfusion:
        self.perfusion_idx = np.array(
            np.where(gtab.bvals <= self.split_b_D)).squeeze()
        self.perfusion_gtab = gradient_table(
            self.gtab.bvals[self.perfusion_idx],
            self.gtab.bvecs[self.perfusion_idx])

        self.perfusion_model = TensorModel(self.perfusion_gtab)
        
        # We'll need a "vanilla" IVIM model:
        self.ivim_model = IvimModel(self.gtab)
        # How many threads in parallel execution:
        self.n_threads = n_threads

    def model_eq1(self, b, *params):
        """
        The model with a fixed perfusion fraction
        """
        bvecs = self.gtab.bvecs
        beta = self._ivim_pf
        Q, Q_star = _reconstruct_tensors(params)
        return _ivim_tensor_equation(beta, b, bvecs, Q_star, Q)

    def model_eq2(self, b, *params):
        """
        The full model, including perfusion fraction as free parameter
        """
        beta = params[0]
        bvecs = self.gtab.bvecs
        Q, Q_star = _reconstruct_tensors(params[1:])
        return _ivim_tensor_equation(beta, b, bvecs, Q_star, Q)

    def _inner_loop(self, vox_chunk):
        model_params = np.zeros((vox_chunk.shape[0], 13))        
        for ii, vox in enumerate(vox_chunk):
            # Extract initial guess of Euler angles for the diffusion fit:
            dt_evecs = self.diffusion_fit.evecs[vox]
            angles_dti = calc_euler(dt_evecs)
            # Extract initial guess of Euler angles for the perfusion fit:
            perfusion_evecs = self.perfusion_fit.evecs[vox]
            angles_perfusion = calc_euler(perfusion_evecs)
            # Initial guess of perfusion fraction based on "vanilla" IVIM:
            self._ivim_pf = np.clip(np.min([self.ivim_fit.perfusion_fraction[vox],
                                    1-self.ivim_fit.perfusion_fraction[vox]]), 0, 1)
            # If diffusivity is lower than this, it's not perfusion!
            min_D_star = 0.003            
            # Put together initial guess for 13 parameters of full model:
            initial = [
                self._ivim_pf,
                np.min([self.diffusion_fit.evals[vox, 0], min_D_star]),
                np.min([self.diffusion_fit.evals[vox, 1], min_D_star]),
                np.min([self.diffusion_fit.evals[vox, 2], min_D_star]),
                angles_dti[0],
                angles_dti[1],
                angles_dti[2],
                np.max([self.perfusion_fit.evals[vox, 0], min_D_star]),
                np.max([self.perfusion_fit.evals[vox, 1], min_D_star]),
                np.max([self.perfusion_fit.evals[vox, 2], min_D_star]), 
                angles_perfusion[0],
                angles_perfusion[1],
                angles_perfusion[2]]

            # Bounds on the parameters:
            lb = (0,
                  0, 0, 0,
                  -np.pi, -np.pi, -np.pi,
                  0.003, 0.003, 0.003,
                  -np.pi, -np.pi, -np.pi)
            ub = (0.5,
                  0.003, 0.003, 0.003,
                  np.pi, np.pi, np.pi,
                  np.inf, np.inf, np.inf,
                  np.pi, np.pi, np.pi)

            # Fit the full model to the data with initial guess and bounds
            try:
                popt, pcov = curve_fit(
                    self.model_eq2,
                    self.gtab.bvals,
                    self.mask_data[vox]/np.mean(self.mask_data[vox, self.gtab.b0s_mask]),
                    p0=initial, bounds=(lb, ub),
                    xtol=0.05,
                    ftol=0.05,
                    maxfev=10000)
            # Sometimes it can't fit the data:
            except RuntimeError:
                popt = np.ones(len(initial)) * np.nan
            model_params[ii] = popt

        return model_params

    def fit(self, data, mask=None):
        """
        Fit the IVIM tensor model
        """
        if mask is None:
            mask = np.ones(data.shape[:-1], dtype=bool)
        self.mask_data = data[mask]
        
        # Fit diffusion tensor to diffusion-weighted data:
        diffusion_data = self.mask_data[:, self.diffusion_idx]
        self.diffusion_fit = self.diffusion_model.fit(diffusion_data)
        # Fit "vanilla" IVIM to all of the data:
        self.ivim_fit = self.ivim_model.fit(self.mask_data)
        # Fit perfusion tensor to perfusion-weighted data:
        perfusion_data = self.mask_data[:, self.perfusion_idx]
        self.perfusion_fit = self.perfusion_model.fit(perfusion_data)
        # Pre-allocate parameters
        #model_params = np.zeros((self.mask_data.shape[0], 13))
        
        voxel_indices = np.arange(self.mask_data.shape[0])
        
        if self.n_threads > 1:
            # Loop over voxels:
            vox_chunks = np.array_split(voxel_indices, self.n_threads)

            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                loop = asyncio.new_event_loop()

                tasks = [
                    loop.run_in_executor(
                        executor,
                        self._inner_loop,
                        vox_chunk,
                    )
                    for vox_chunk in vox_chunks
                ]
                
                try:
                    model_params = np.concatenate(list(tqdm(loop.run_until_complete(asyncio.gather(*tasks)))))
                finally:
                    loop.close()
        else: 
            model_params = self._inner_loop(voxel_indices)
            
        return IvimTensorFit(self, model_params)


class IvimTensorFit(ReconstFit):

    def __init__(self, model, model_params):
        self.model = model
        self.model_params = model_params
        perfusion_params = np.zeros((self.model_params.shape[0], 12))
        diffusion_params = np.zeros((self.model_params.shape[0], 12))
        self.perfusion_fraction = np.zeros(self.model_params.shape[0])
        for vox in range(self.model_params.shape[0]):
            self.perfusion_fraction[vox] = self.model_params[vox, 0]
            tensor_evecs, tensor_evals = _reconstruct_tensor(
                *self.model_params[vox, 1:7])
            diffusion_params[vox] = np.hstack(
                [tensor_evals, tensor_evecs.ravel()])
            perfusion_evecs, perfusion_evals = _reconstruct_tensor(
                *self.model_params[vox, 7:])
            perfusion_params[vox] = np.hstack(
                [perfusion_evals, perfusion_evecs.ravel()])

        self.diffusion_fit = TensorFit(
            self.model.diffusion_model, diffusion_params)
        self.perfusion_fit = TensorFit(
            self.model.perfusion_model, perfusion_params)

    def predict(self, gtab, s0=1):
        bvecs = gtab.bvecs
        b = gtab.bvals
        prediction = np.zeros((self.model_params.shape[0], gtab.bvals.shape[0]))
        for vox in range(self.model_params.shape[0]):
            Q = self.diffusion_fit.quadratic_form[vox]
            Q_star = self.perfusion_fit.quadratic_form[vox]
            prediction[vox] = _ivim_tensor_equation(
                self.perfusion_fraction[vox], b, bvecs, Q_star, Q)
        return prediction
