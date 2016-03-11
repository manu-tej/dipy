""" Classes and functions for fitting tensors for CHARMED """
from __future__ import division, print_function, absolute_import

import warnings

import numpy as np

import scipy.optimize as opt
from .base import ReconstModel

from dipy.reconst.dti import (TensorFit, design_matrix, _min_positive_signal,
                              decompose_tensor, from_lower_triangular,
                              lower_triangular, apparent_diffusion_coef)
from dipy.reconst.dki import _positive_evals

from dipy.core.sphere import Sphere
from .vec_val_sum import vec_val_vect
from dipy.core.ndindex import ndindex

def charmed_prediction(params, gtab, S0 ):
    """
    Predict a signal given tensor parameters.

    Parameters
    ----------
    params : ndarray
        Tensor parameters. The last dimension should have 12 tensor
        parameters: 3 eigenvalues, followed by the 3 corresponding
        eigenvectors.

    gtab : a GradientTable class instance
        The gradient table for this prediction


    Notes
    -----
    The Predicted signal is given by

        E(Q,Delta) = f_H * E_H(Q,Delta) + f_R * E_R(Q,Delta)

        where E_H and E_R are signals arising from hindered and restricted
        components.

        Qper2_H = gtab.qvals**2*(1-(sin(Q(:,1))*sin(theta_H)*cos(Q(:,2)-phi_H)+cos(Q(:,1))*cos(theta_H))**2)

        Qpar2_H = gtab.qvals**2*(sin(Q(:,1))*sin(theta_H)*cos(Q(:,2)-phi_H)+cos(Q(:,1))*cos(theta_H))**2;

        Qper2_R = gtab.qvals**2*(1-(sin(Q(:,1))*sin(theta_R)*cos(Q(:,2)-phi_R)+cos(Q(:,1))*cos(theta_R))**2);

        Qpar2_R = gtab.qvals**2*(sin(Q(:,1))*sin(theta_R)*cos(Q(:,2)-phi_R)+cos(Q(:,1))*cos(theta_R))**2;


        E_H = e^(-4 * pi^2 * (Delta - (delta/3)) * (Qper2_H *lambda_per + Qpar2_H * lambda_par))
        E_R = e^(-4 * pi^2 * (Qpar2_R * (Delta - (delta/3)) * Dif_par - ((R^4 * Qper2_R)/Dif_per * Tau)*(2-(99/112)*(R^2/Dif_per* Tau)))

        E = f_R * E_R + f_H * E_H

        Initially data is fitted by DTI to know eigenvalues and eigenvectors of
        Diffusion tensor. The estimated parameters are used to fit the CHARMED
        signal.

        """
        evals = params[..., :3]
        evecs = params[..., 3:-2].reshape(params.shape[:-1] + (3, 3))
        f = params[..., 12]
        S0 = params[..., 13]
        qform = vec_val_vect(evecs, evals)
        sphere = Sphere(xyz=gtab.bvecs[~gtab.b0s_mask])
        adc = apparent_diffusion_coef(qform, sphere)
        mask = _positive_evals(evals[..., 0], evals[..., 1], evals[..., 2])

        # First do the calculation for the diffusion weighted measurements:
        pred_sig = np.zeros(f.shape + (gtab.bvals.shape[0],))
        index = ndindex(f.shape)
        for v in index:
            if mask[v]:
                E_H = f[v] * np.exp(-4 * np.pi**2 * (Delta - (delta/3)) * (Qper2_H * lambda_per + Qpar2_H * lambda_par))
                E_R = (1-f[v]) * np.exp(-4 * np.pi**2 * (Qpar2_R * (Delta - (delta/3)) * Dif_par - ((R^4 * Qper2_R)/Dif_per * Tau) * (2 - (99/112) * (R**2/(Dif_per * Tau)))))
                pre_pred_sig = S0[v] * (E_H + E_R)

                # Then we need to sort out what goes where:
                pred_s = np.zeros(pre_pred_sig.shape[:-1] + (gtab.bvals.shape[0],))

                # These are the diffusion-weighted values
                pred_s[..., ~gtab.b0s_mask] = pre_pred_sig

                # For completeness, we predict the mean S0 for the non-diffusion
                # weighted measurements, which is our best guess:
                pred_s[..., gtab.b0s_mask] = S0[v]
                pred_sig[v] = pred_s

        return pred_sig


class CharmedTensorModel(ReconstModel):
    """ Diffusion Tensor
    """
    def __init__(self, gtab, fit_method="WLS", *args, **kwargs):
        """ A Diffusion Tensor Model [1]_, [2]_.

        Parameters
        ----------
        gtab : GradientTable class instance

        fit_method : str or callable
            str can be one of the following:
            'WLS' for weighted least squares
                dti.wls_fit_tensor
            'LS' or 'OLS' for ordinary least squares
                dti.ols_fit_tensor
            'NLLS' for non-linear least-squares
                dti.nlls_fit_tensor
            'RT' or 'restore' or 'RESTORE' for RESTORE robust tensor
                fitting [3]_
                dti.restore_fit_tensor

            callable has to have the signature:
              fit_method(design_matrix, data, *args, **kwargs)

        args, kwargs : arguments and key-word arguments passed to the
           fit_method. See dti.wls_fit_tensor, dti.ols_fit_tensor for details

        min_signal : float
            The minimum signal value. Needs to be a strictly positive
            number. Default: minimal signal in the data provided to `fit`.

        References
        ----------
        To be done
        """
        ReconstModel.__init__(self, gtab)

        if not callable(fit_method):
            try:
                fit_method = common_fit_methods[fit_method]
            except KeyError:
                e_s = '"' + str(fit_method) + '" is not a known fit '
                e_s += 'method, the fit method should either be a '
                e_s += 'function or one of the common fit methods'
                raise ValueError(e_s)
        self.fit_method = fit_method
        self.design_matrix = design_matrix(self.gtab)
        self.args = args
        self.kwargs = kwargs
        self.min_signal = self.kwargs.pop('min_signal', None)
        if self.min_signal is not None and self.min_signal <= 0:
            e_s = "The `min_signal` key-word argument needs to be strictly"
            e_s += " positive."
            raise ValueError(e_s)

    def fit(self, data, mask=None):
        """ Fit method of the DTI model class

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[:-1]

        """
        if mask is None:
            # Flatten it to 2D either way:
            data_in_mask = np.reshape(data, (-1, data.shape[-1]))
        else:
            # Check for valid shape of the mask
            if mask.shape != data.shape[:-1]:
                raise ValueError("Mask is not the same shape as data.")
            mask = np.array(mask, dtype=bool, copy=False)
            data_in_mask = np.reshape(data[mask], (-1, data.shape[-1]))

        if self.min_signal is None:
            min_signal = _min_positive_signal(data)
        else:
            min_signal = self.min_signal

        data_in_mask = np.maximum(data_in_mask, min_signal)
        params_in_mask = self.fit_method(self.design_matrix, data_in_mask,
                                         *self.args, **self.kwargs)

        if mask is None:
            out_shape = data.shape[:-1] + (-1, )
            params = params_in_mask.reshape(out_shape)
        else:
            params = np.zeros(data.shape[:-1] + (12,))
            params[mask, :] = params_in_mask

        return CharmedTensorFit(self, params)

    def predict(self, params, S0=1):
        """
        Predict a signal for this TensorModel class instance given parameters.

        Parameters
        ----------
        params : ndarray
            The last dimension should have 12 tensor parameters: 3
            eigenvalues, followed by the 3 eigenvectors

        S0 : float or ndarray
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1
        """
        return charmed_prediction(params, self.gtab, S0)

class CharmedTensorFit(TensorFit):
    """ Class for fitting the CHARMED Model """
    def __init__(self, model, model_params):
        """ Initialize a CharmedTensorFit class instance.
        Since the charmed model is an extension of DTI, class
        instance is defined as subclass of the TensorFit from dti.py

        Parameters
        ----------
        model : CharmedTensorModel Class instance
            Class instance containing the charmed model for the fit
        model_params : ndarray (x, y, z, 14) or (n, 14)
            All parameters estimated from the charmed model.
            Parameters are ordered as follows:
                1) Three diffusion tensor's eigenvalues of hindered and
                   restricted parts
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) The volume fractions of the hindered and restricted
                   compartments
                4) Parallel Diffusivity in restricted compartments
                5) Orientations of restricted part in spherical co-ordinates
                """
        TensorFit.__init__(self, model, model_params)

    @property
    def f(self):
        """ Returns the free water diffusion volume fraction f """
        return self.model_params[..., 12]

    @property
    def S0(self):
        """ Returns the non-diffusion weighted signal estimate """
        return self.model_params[..., 13]

    def predict(self, gtab, step=None):
        r""" Given a charmed model fit, predict the signal on the
        vertices of a gradient table

        Parameters
        ----------
        gtab : a GradientTable class instance
            The gradient table for this prediction

        step : int, optional
            Number of voxels to be processed simultaneously
        """
        shape = self.model_params.shape[:-1]
        size = np.prod(shape)
        if step is None:
            step = self.model.kwargs.get('step', size)
        if step >= size:
            return fwdti_prediction(self.model_params, gtab)
        params = np.reshape(self.model_params,
                            (-1, self.model_params.shape[-1]))
        predict = np.empty((size, gtab.bvals.shape[0]))
        for i in range(0, size, step):
            predict[i:i+step] = fwdti_prediction(params[i:i+step], gtab)
        return predict.reshape(shape + (gtab.bvals.shape[0], ))
