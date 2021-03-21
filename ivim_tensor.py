import numpy as np
from scipy.optimize import curve_fit
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import lower_triangular, from_lower_triangular, vec_val_vect
from dipy.core.geometry import sphere2cart, cart2sphere
from dipy.reconst.base import ReconstFit, ReconstModel
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.dti import TensorModel, TensorFit, decompose_tensor, from_lower_triangular


def _ivim_tensor_equation(beta, b, bvecs, Q_star, Q):
    return beta * np.exp(-b * np.diag(bvecs @ Q_star @ bvecs.T)) + (1 - beta) * np.exp(-b * np.diag(bvecs @ Q @ bvecs.T))

def _reconstruct_tensor(eval0, eval1, eval2, theta0, phi0, theta1, phi1, theta2, phi2):
    """ 
    Recontruct a quadratic form from a series of eigenvalues and rotations 
    """
    evecs = np.vstack([np.array(sphere2cart(1, theta0, phi0)),
                       np.array(sphere2cart(1, theta1, phi1)),
                       np.array(sphere2cart(1, theta2, phi2))])
    evals = np.array([eval0, eval1, eval2])
    return vec_val_vect(evecs, evals)

def _reconstruct_tensors(params):
    Q = _reconstruct_tensor(params[0], params[1], params[2], # DTI eigenvalues
                            params[3], params[4], # DTI evec0 rotations
                            params[5], params[6], # DTI evec1 rotations
                            params[7], params[8], # DTI evec2 rotations
                           )
    Q_star = _reconstruct_tensor(params[9], params[10], params[11], # DTI eigenvalues
                                 params[12], params[13], # DTI evec0 rotations
                                 params[14], params[15], # DTI evec1 rotations
                                 params[16], params[17], # DTI evec2 rotations
                                 )
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
        # Fit separate tensors for data split:
        diffusion_data = data[self.diffusion_idx]
        self.diffusion_fit = self.diffusion_model.fit(diffusion_data, mask)
        
        theta0, phi0 = cart2sphere(*self.diffusion_fit.evecs[0])[1:]
        theta1, phi1 = cart2sphere(*self.diffusion_fit.evecs[1])[1:]
        theta2, phi2 = cart2sphere(*self.diffusion_fit.evecs[2])[1:]
    
        perfusion_data = data[self.perfusion_idx]
        self.perfusion_fit = self.perfusion_model.fit(perfusion_data, mask)

        theta_star0, phi_star0 = cart2sphere(*self.perfusion_fit.evecs[0])[1:]
        theta_star1, phi_star1 = cart2sphere(*self.perfusion_fit.evecs[1])[1:]
        theta_star2, phi_star2 = cart2sphere(*self.perfusion_fit.evecs[2])[1:]
    
        fractions_for_probe = np.arange(0, 0.5, 0.05)
        self.fits = np.zeros((fractions_for_probe.shape[0], 18))
        self.errs = np.zeros(fractions_for_probe.shape[0])
        self.beta = np.zeros(fractions_for_probe.shape[0])
        initial = np.hstack([self.diffusion_fit.evals[0], 
                             self.diffusion_fit.evals[1], 
                             self.diffusion_fit.evals[2], 
                             theta0, phi0,
                             theta1, phi1,
                             theta2, phi2,
                             self.perfusion_fit.evals[0], 
                             self.diffusion_fit.evals[1], 
                             self.diffusion_fit.evals[2], 
                             theta_star0, phi_star0,
                             theta_star1, phi_star1,
                             theta_star2, phi_star2,
                            ])
        lb = (0, 0, 0, 0, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, 0, 0, 0, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi) 
        ub = (1, np.inf, np.inf, np.inf, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.inf, np.inf, np.inf, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi) 
        
        # Instead of estimating perfusion_fraction directly, we start by finding 
        # a perfusion fraction that works for the other parameters
        for ii, perfusion_fraction in enumerate(fractions_for_probe):
            self.perfusion_fraction = perfusion_fraction
            try:
                popt, pcov = curve_fit(self.model_eq1,  self.gtab.bvals, data, p0=initial, 
                                      bounds=(lb[1:], ub[1:]))
                err = np.sum(np.power(self.model_eq1(self.gtab.bvals, *popt) - data, 2))
                self.fits[ii] = popt
                self.errs[ii] = err
                self.beta[ii] = perfusion_fraction
            except RuntimeError: 
                self.fits[ii] = np.nan
                self.errs[ii] = np.pi
                self.beta[ii] = np.nan
                
        min_err = np.argmin(self.errs)
        initial = np.hstack([self.beta[min_err], self.fits[min_err]])
        
        popt, pcov = curve_fit(self.model_eq2,  self.gtab.bvals, data, p0=initial, bounds=(lb, ub))
        return IvimTensorFit(self, popt)
        
                            
class IvimTensorFit(ReconstFit):
    # XXX Still need to adapt to new model_params!
    
    def __init__(self, model, model_params):
        self.model = model
        self.model_params = model_params
        tensor_evals, tensor_evecs = decompose_tensor(from_lower_triangular(self.model_params[1:7]))
        tensor_params = np.hstack([tensor_evals, tensor_evecs.ravel()])
        perfusion_evals, perfusion_evecs = decompose_tensor(from_lower_triangular(self.model_params[7:]))
        perfusion_params = np.hstack([perfusion_evals, perfusion_evecs.ravel()])
        self.diffusion_fit = TensorFit(self.model.diffusion_model, tensor_params)
        self.perfusion_fit = TensorFit(self.model.perfusion_model, perfusion_params)
        self.perfusion_fraction = np.min([model_params[0], 1 - model_params[0]])