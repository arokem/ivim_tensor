import numpy as np
from scipy.optimize import curve_fit
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import lower_triangular, from_lower_triangular, vec_val_vect
from dipy.core.geometry import sphere2cart, cart2sphere, euler_matrix, vec2vec_rotmat, decompose_matrix, rodrigues_axis_rotation
from dipy.reconst.base import ReconstFit, ReconstModel
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.dti import TensorModel, TensorFit, decompose_tensor, from_lower_triangular
from dipy.reconst.ivim import IvimModel


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
        
        self.perfusion_idx = np.where(gtab.bvals <= self.split_b_D)    
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
        ivim_fit = self.ivim_model.fit(data)
        
        # Fit separate tensors for data split:
        diffusion_data = data[self.diffusion_idx]
        self.diffusion_fit = self.diffusion_model.fit(diffusion_data, mask)
        
        # Calculate Euler angles for the diffusion fit:
        # We start by calculating the rotation matrix that will align 
        # the first eigenvector
        rot0 = np.eye(4)
        rot0[:3, :3] = vec2vec_rotmat(self.diffusion_fit.evecs[0], np.eye(3)[0])
        scale, shear, angles0, translate, perspective = decompose_matrix(rot0)
        em = euler_matrix(*angles0)
        # Now, we need another rotation to bring the second eigenvector to the right 
        # direction
        ang1 = np.arccos(np.dot(self.diffusion_fit.evecs[1], em[1, :3]) / 
                         (np.linalg.norm(self.diffusion_fit.evecs[1]) * np.linalg.norm(em[1, :3])))
        rar = np.eye(4)
        rar[:3, :3] = rodrigues_axis_rotation(self.diffusion_fit.evecs[0], np.rad2deg(ang1))
        
        # We combine these two rotations and decompose the combined matrix to give us 
        # three Euler angles, which will be our parameters
        scale, shear, angles_dti, translate, perspective = decompose_matrix(em @ rar)
            
        perfusion_data = data[self.perfusion_idx]
        self.perfusion_fit = self.perfusion_model.fit(perfusion_data, mask)

        # Calculate Euler angles for the perfusion fit:
        rot0 = np.eye(4)
        rot0[:3, :3] = vec2vec_rotmat(self.perfusion_fit.evecs[0], np.eye(3)[0])
        scale, shear, angles0, translate, perspective = decompose_matrix(rot0)
        em = euler_matrix(*angles0)
        ang1 = np.arccos(np.dot(self.perfusion_fit.evecs[1], em[1, :3]) / 
                         (np.linalg.norm(self.perfusion_fit.evecs[1]) * np.linalg.norm(em[1, :3])))
        rar = np.eye(4)
        rar[:3, :3] = rodrigues_axis_rotation(self.perfusion_fit.evecs[0], np.rad2deg(ang1))
        scale, shear, angles_perfusion, translate, perspective = decompose_matrix(em @ rar)

        self.perfusion_fraction = ivim_fit.perfusion_fraction 
        # XXX TODO: Need to make sure that evals are sorted 
        # so that largest is first *after the fitting*
        initial = [self.diffusion_fit.evals[0], 
                   self.diffusion_fit.evals[1], 
                   self.diffusion_fit.evals[2], 
                   angles_dti[0],
                   angles_dti[1],
                   angles_dti[2],
                   self.perfusion_fit.evals[0], 
                   self.perfusion_fit.evals[1], 
                   self.perfusion_fit.evals[2], 
                   angles_perfusion[0],
                   angles_perfusion[1],
                   angles_perfusion[2]                  ]
        
        lb = (0, 0, 0, -np.pi, -np.pi, -np.pi, 0, 0, 0, -np.pi, -np.pi, -np.pi)
        ub = (0.004, 0.004, 0.004, np.pi, np.pi, np.pi, 0.2, 0.2, 0.2, np.pi, np.pi, np.pi)
        popt, pcov = curve_fit(self.model_eq1,
                               self.gtab.bvals, 
                               data/np.mean(data[self.gtab.b0s_mask]), 
                               p0=initial, bounds=(lb, ub))
        return IvimTensorFit(self, popt)
        
                            
class IvimTensorFit(ReconstFit):
    
    def __init__(self, model, model_params):
        self.model = model
        self.model_params = model_params
        tensor_evecs, tensor_evals = _reconstruct_tensor(*self.model_params[:6])        
        tensor_params = np.hstack([tensor_evals, tensor_evecs.ravel()])
        perfusion_evecs, perfusion_evals = _reconstruct_tensor(*self.model_params[6:])
        perfusion_params = np.hstack([perfusion_evals, perfusion_evecs.ravel()])
        self.diffusion_fit = TensorFit(self.model.diffusion_model, tensor_params)
        self.perfusion_fit = TensorFit(self.model.perfusion_model, perfusion_params)
        self.perfusion_fraction = self.model.perfusion_fraction
    
    def predict(self, gtab, s0=1):
        bvecs = gtab.bvecs
        b = gtab.bvals
        Q = self.diffusion_fit.quadratic_form
        Q_star = self.perfusion_fit.quadratic_form
        return _ivim_tensor_equation(self.perfusion_fraction, b, bvecs, Q_star, Q)
