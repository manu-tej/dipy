import numpy as np
from scipy.optimize import curve_fit, leastsq
from dipy.reconst import dti
import numpy as np
import dipy.data as dpd
import dipy.core.gradients as dpg	
from dipy.segment.mask import median_otsu
import nibabel as nib
from dipy.core.gradients import GradientTable


fdata, fbvals, fbvecs = dpd.get_data('small_101D')
img = nib.load(fdata)
data = img.get_data()
big_delta=150
small_delta=40
gtab = dpg.gradient_table(fbvals, fbvecs, big_delta=150,
					  small_delta=40, b0_threshold=1000)
a = GradientTable(gtab.gradients, big_delta=150,
				  small_delta=40, b0_threshold=1000)
a.bvals = gtab.bvals[gtab.b0s_mask]
a.bvecs = gtab.bvecs[gtab.b0s_mask]
a.gradients = gtab.gradients[gtab.b0s_mask]
a.b0s_mask = gtab.b0s_mask[gtab.b0s_mask]
	
maskdata, mask = median_otsu(data, 3, 1, True,
                             vol_idx=range(10, 50), dilate=2)
print('maskdata.shape (%d, %d, %d, %d)' % maskdata.shape)


def intial_conditions_prediction(a, maskdata):

	tenmodel = dti.TensorModel(a)
	tenfit = tenmodel.fit(maskdata[:,:,:,gtab.b0s_mask])

	intial_params  = {}	# intialising a dictionary
						# for storing intial parameters

	intial_params['lambda_per'] = (dti.axial_diffusivity(tenfit.evals))
							# lambda_per axial diffusivity
	print(intial_params['lambda_per'].shape)  
	print('p')
	intial_params['lambda_par'] = (dti.radial_diffusivity(tenfit.evals))
							#  lambda_par is radial 
							#  diffusivity

	return intial_params

def create_qtable(a, origin=np.array([0])):
    """ create a normalized version of gradients

    Parameters
    ----------
    gtab : GradientTable
    origin : (3,) ndarray
        center of qspace

    Returns
    -------
    qtable : ndarray
    """

    bv = a.bvals
    bsorted = np.sort(bv[np.bitwise_not(~a.b0s_mask)])
    for i in range(len(bsorted)):
        bmin = bsorted[i]
        try:
            if np.sqrt(bv.max() / bmin) > origin + 1:
                continue
            else:
                break
        except ZeroDivisionError:
            continue

    bv = np.sqrt(bv / bmin)
    qtable = np.vstack((bv, bv, bv)).T * a.bvecs
    return qtable

def hindered_signal(gtab, angles):

	theta_Q = np.arctan(qvec_H[1]/qvec_H[0])
	phi_Q = np.sqrt(qvec_H[1]**2 + qvec_H[0]**2)
	phi_Q = np.arctan(phi_Q/qvec_H[2])

	intial_params = intial_conditions_prediction(a,maskdata)

	Qper2_H = (a.qvals**2)*(1-(np.sin(theta_Q)*np.sin(angles[1])*np.cos(phi_Q - angles[0])+np.cos(theta_Q)*np.cos(angles[1]))**2)
	Qpar2_H = (a.qvals**2)*((np.sin(theta_Q))*np.sin(angles[1])*np.cos(phi_Q - angles[0])+np.cos(theta_Q)*np.cos(angles[1]))**2
	E_H = np.exp(-4 * np.pi**2 * (big_delta - (small_delta/3)) * (Qper2_H * intial_params['lambda_per'] + 
    			 										Qpar2_H * intial_params['lambda_par']))
	return E_H

def hingered_residual(angles, data, gtab):
	return data - hindered_signal(gtab,angles)

def hindered_fit(maskdata, gtab):
	qvec_H = create_qtable(a)
	k = qvec_H
	print(qvec_H[1])

	ydata = maskdata[:,:,:, gtab.b0s_mask]
	print(ydata.ravel().shape)
	while (qvec_H[:,1].shape != ydata.ravel().shape):
		qvec_H = np.concatenate((qvec_H,k), axis=0)
		a.qvals = np.concatenate((a.qvals, gtab.qvals[gtab.b0s_mask]), axis=0)
	xdata = np.transpose(qvec_H)
	print(xdata.shape)
	print(ydata.ravel().shape)
	param,pov = curve_fit(hindered_signal, xdata, ydata.ravel())
	return param

def hindered_and_restricted_signal(gtab, params):


	theta_Q = np.arctan(qvec[1]/qvec[0])
	phi_Q = np.sqrt(qvec[1]**2 + qvec[0]**2)
	phi_Q = np.arctan(phi_Q/qvec[2])

	Qper2_H = (a.qvals**2)*(1-(np.sin(theta_Q)*np.sin(params[1])*np.cos(phi_Q - params[0])+np.cos(theta_Q)*np.cos(params[1]))**2)
	Qpar2_H = (a.qvals**2)*((np.sin(theta_Q))*np.sin(params[1])*np.cos(phi_Q - params[0])+np.cos(theta_Q)*np.cos(params[1]))**2

	Qper2_R = (a.qvals**2)*(1-(np.sin(theta_Q)*np.sin(params[2])*np.cos(phi_Q - params[3])+np.cos(theta_Q)*np.cos(parmas[2]))**2)
	Qpar2_R = (a.qvals**2)*((np.sin(theta_Q))*np.sin(params[2])*np.cos(phi_Q - params[3])+np.cos(theta_Q)*np.cos(params[2]))**2

	E_H = np.exp(-4 * np.pi**2 * (big_delta - (small_delta/3)) * (Qper2_R * params[4] + 
    			 										Qpar2_R * params[5]))
	E_R = (1-f[v]) * np.exp(-4 * np.pi**2 * (Qpar2_R * (big_delta - (small_delta/3)) * params[6] - ((R^4 * Qper2_R)/Dif_per * Tau) * (2 - (99/112) * (R**2/(Dif_per * Tau)))))

	return E_R + E_H


def hind_and_rest_residual(params, data, gtab):
	return data - hindered_and_restricted_signal(gtab, params)

def hind_and_rest_fit(maskdata, gtab, x0):
	charmed_params, flag = leastsq(hind_and_rest_residual, x0, args=(maskdata, gtab))
	return charmed_params

def noise_function(E_est, noise):
	E = np.sqrt(E_est**2 + noise**2)
	return E

def noise_residual(noise, data, E_est):
	return data - noise_function(E_est,noise)

def noise_fit(data, E_est, n0):
	noise_param , flag = leastsq(noise_residual, n0, args=(data,E_est))
	return noise_param



intial_params = intial_conditions_prediction(a, maskdata)
"""
print(intial_params['lambda_per'])
print(maskdata[:,:,:,gtab.b0s_mask].ravel().shape)
hind_param = hindered_fit(maskdata,gtab)
print(hind_param)
"""
