import numpy as np
from scipy.optimize import curve_fit
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import lower_triangular, from_lower_triangular
from dipy.reconst.base import ReconstFit, ReconstModel
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.dti import TensorModel, TensorFit, decompose_tensor, from_lower_triangular

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
        theta = self.gtab.bvecs
        beta = self.perfusion_fraction
        Q = from_lower_triangular(np.array(params[:6]))
        Q_star = from_lower_triangular(np.array(params[6:]))
        return beta * np.exp(-b * np.diag(theta @ Q_star @ theta.T)) + (1 - beta) * np.exp(-b * np.diag(theta @ Q @ theta.T))

    def model_eq2(self, b, *params): 
        """ 
        The full model, including perfusion fraction
        """
        theta = self.gtab.bvecs
        beta = params[0]
        Q = from_lower_triangular(np.array(params[1:7]))
        Q_star = from_lower_triangular(np.array(params[7:]))
        return beta * np.exp(-b * np.diag(theta @ Q_star @ theta.T)) + (1 - beta) * np.exp(-b * np.diag(theta @ Q @ theta.T))

    
    def fit(self, data, mask=None):
        """ 
        For now, we assume that data is from a single voxel, and we'll generalize later
        """
        # Fit separate tensors for data split:
        diffusion_data = data[self.diffusion_idx]
        self.diffusion_fit = self.diffusion_model.fit(diffusion_data, mask)
        self.q_initial = lower_triangular(self.diffusion_fit.quadratic_form)
        perfusion_data = data[self.perfusion_idx]
        self.perfusion_fit = self.perfusion_model.fit(perfusion_data, mask)
        self.q_star_initial = lower_triangular(self.perfusion_fit.quadratic_form).squeeze()

        fractions_for_probe = np.arange(0, 0.5, 0.05)
        self.fits = np.zeros((fractions_for_probe.shape[0], 12))
        self.errs = np.zeros(fractions_for_probe.shape[0])
        self.beta = np.zeros(fractions_for_probe.shape[0])
        initial = np.hstack([self.q_initial, self.q_star_initial])
        # Instead of estimating perfusion_fraction directly, we start by finding 
        # a perfusion fraction that works for the other parameters
        for ii, perfusion_fraction in enumerate(fractions_for_probe):
            self.perfusion_fraction = perfusion_fraction
            try:
                popt, pcov = curve_fit(self.model_eq1,  self.gtab.bvals, data, p0=initial)
                err = np.sum(np.power(self.model_eq1(self.gtab.bvals, *popt) - data, 2))
                self.fits[ii] = popt
                self.errs[ii] = err
                self.beta[ii] = perfusion_fraction
            except RuntimeError: 
                self.fits[ii] = np.nan
                self.errs[ii] = np.inf
                self.beta[ii] = np.nan
                
        min_err = np.argmin(self.errs)
        initial = np.hstack([self.beta[min_err], self.fits[min_err]])
        popt, pcov = curve_fit(self.model_eq2,  self.gtab.bvals, data, p0=initial, 
        bounds=((0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf),                     (1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)))
        return IvimTensorFit(self, popt)
        
                            
class IvimTensorFit(ReconstFit):
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