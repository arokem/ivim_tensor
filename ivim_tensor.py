import numpy as np
from scipy.optimize import curve_fit
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import lower_triangular, from_lower_triangular, vec_val_vect
from dipy.core.geometry import sphere2cart, cart2sphere, euler_matrix, vec2vec_rotmat, decompose_matrix, rodrigues_axis_rotation
from dipy.reconst.base import ReconstFit, ReconstModel
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.dti import TensorModel, TensorFit, decompose_tensor, from_lower_triangular
from dipy.reconst.ivim import IvimModel
from tqdm import tqdm


def _ivim_tensor_equation(beta, b, bvecs, Q_star, Q):
    return beta * np.exp(-b * np.diag(bvecs @ Q_star @ bvecs.T)) + (1 - beta) * np.exp(-b * np.diag(bvecs @ Q @ bvecs.T))

def _reconstruct_tensor(eval0, eval1, eval2, eul0, eul1, eul2):
    """ 
    Reconstruct a quadratic form from a series of eigenvalues and Euler rotations 
    """
    R = euler_matrix(eul0, eul1, eul2)    
    evecs = R[:3, :3]
    evals = np.array([eval0, eval1, eval2])
    return evecs, evals

def _reconstruct_tensors(params):
    Q = vec_val_vect(*_reconstruct_tensor(
        params[0], params[1], params[2],
        params[3], params[4], params[5]))
    
    Q_star = vec_val_vect(*_reconstruct_tensor(
        params[6], params[7], params[8],
        params[9], params[10], params[11]))
    
    return Q, Q_star

def calc_euler(evecs):
    rot0 = np.eye(4)
    rot0[:3, :3] = vec2vec_rotmat(evecs[0], np.eye(3)[0])
    scale, shear, angles0, translate, perspective = decompose_matrix(rot0)
    em = euler_matrix(*angles0)
    # Now, we need another rotation to bring the second eigenvector to the right 
    # direction
    ang1 = np.arccos(np.dot(evecs[1], em[1, :3]) / 
                     (np.linalg.norm(evecs[1]) * np.linalg.norm(em[1, :3])))
    rar = np.eye(4)
    rar[:3, :3] = rodrigues_axis_rotation(evecs[0], np.rad2deg(ang1))

    # We combine these two rotations and decompose the combined matrix to give us 
    # three Euler angles, which will be our parameters
    scale, shear, angles, translate, perspective = decompose_matrix(em @ rar)
    return angles


class IvimTensorModel(ReconstModel):
    def __init__(self, gtab, split_b_D=200.0, bounds=[]):
        ReconstModel.__init__(self, gtab)
        self.split_b_D = split_b_D
        self.bounds = bounds
        # Use two separate tensors for initial estimation:
        self.diffusion_idx = np.hstack([np.where(gtab.bvals > self.split_b_D),
                                        np.where(gtab.b0s_mask)]).squeeze()

        self.diffusion_gtab = gradient_table(self.gtab.bvals[self.diffusion_idx],
                                             self.gtab.bvecs[self.diffusion_idx])
        
        self.diffusion_model = TensorModel(self.diffusion_gtab)
        
        self.perfusion_idx = np.array(np.where(gtab.bvals <= self.split_b_D)).squeeze()
        self.perfusion_gtab = gradient_table(self.gtab.bvals[self.perfusion_idx],
                                             self.gtab.bvecs[self.perfusion_idx])
        
        self.perfusion_model = TensorModel(self.perfusion_gtab)
    
        self.ivim_model = IvimModel(self.gtab)
        
        
    def model_eq1(self, b, *params): 
        """ 
        The model with fixed perfusion fraction
        """
        bvecs = self.gtab.bvecs
        beta = self.perfusion_fraction
        Q, Q_star = _reconstruct_tensors(params)
        return _ivim_tensor_equation(beta, b, bvecs, Q_star, Q)

    def model_eq2(self, b, *params): 
        """ 
        The full model, including perfusion fraction
        """
        beta = params[0]
        bvecs = self.gtab.bvecs
        Q, Q_star = _reconstruct_tensors(params[1:])
        return _ivim_tensor_equation(beta, b, bvecs, Q_star, Q)
    
    def fit(self, data, mask=None):
        """ 
        For now, we assume that data is from a single voxel, and we'll generalize later
        """
        if mask is None:
            mask = np.ones(data.shape[:-1], dtype=bool)
        mask_data = data[mask]
        diffusion_data = mask_data[:, self.diffusion_idx]        
        self.diffusion_fit = self.diffusion_model.fit(diffusion_data)
        self.ivim_fit = self.ivim_model.fit(mask_data)
        perfusion_data = mask_data[:, self.perfusion_idx]
        self.perfusion_fit = self.perfusion_model.fit(perfusion_data)
        model_params = np.zeros((mask_data.shape[0], 13))
        for vox in tqdm(range(mask_data.shape[0])):
            # Calculate Euler angles for the diffusion fit:
            # We start by calculating the rotation matrix that will align 
            # the first eigenvector
            dt_evecs = self.diffusion_fit.evecs[vox]
            angles_dti = calc_euler(dt_evecs)
            perfusion_evecs = self.perfusion_fit.evecs[vox]
            angles_perfusion = calc_euler(perfusion_evecs)
            ivim_pf = np.max([ivim_fit.perfusion_fraction[vox], 0.001])
            initial = [
                ivim_pf,
                self.diffusion_fit.evals[vox, 0], 
                self.diffusion_fit.evals[vox, 1], 
                self.diffusion_fit.evals[vox, 2], 
                angles_dti[0],
                angles_dti[1],
                angles_dti[2],
                self.perfusion_fit.evals[vox, 0], 
                self.perfusion_fit.evals[vox, 1], 
                self.perfusion_fit.evals[vox, 2], 
                angles_perfusion[0],
                angles_perfusion[1],
                angles_perfusion[2]]

            lb = (0, 0, 0, 0, -np.pi, -np.pi, -np.pi, 0, 0, 0, -np.pi, -np.pi, -np.pi)
            ub = (ivim_pf, 0.004, 0.004, 0.004, np.pi, np.pi, np.pi, 0.2, 0.2, 0.2, np.pi, np.pi, np.pi)
            try:
                popt, pcov = curve_fit(self.model_eq2,
                                       self.gtab.bvals, 
                                       mask_data[vox]/np.mean(mask_data[vox, self.gtab.b0s_mask]), 
                                       p0=initial, bounds=(lb, ub))
            except:
                popt = np.ones(len(initial)) * np.nan
            model_params[vox] = popt
        
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
            tensor_evecs, tensor_evals = _reconstruct_tensor(*self.model_params[vox, 1:7])        
            diffusion_params[vox] = np.hstack([tensor_evals, tensor_evecs.ravel()])
            perfusion_evecs, perfusion_evals = _reconstruct_tensor(*self.model_params[vox, 7:])
            perfusion_params[vox] = np.hstack([perfusion_evals, perfusion_evecs.ravel()])
            
        self.diffusion_fit = TensorFit(self.model.diffusion_model, diffusion_params)
        self.perfusion_fit = TensorFit(self.model.perfusion_model, perfusion_params)
    
    def predict(self, gtab, s0=1):
        bvecs = gtab.bvecs
        b = gtab.bvals
        Q = self.diffusion_fit.quadratic_form
        Q_star = self.perfusion_fit.quadratic_form
        return _ivim_tensor_equation(self.perfusion_fraction, b, bvecs, Q_star, Q)
