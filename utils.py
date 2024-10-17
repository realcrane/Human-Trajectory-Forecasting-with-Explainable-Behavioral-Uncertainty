import numpy as np
import copy
import torch
import os
import random

def calculate_v(x_seq):
    length = x_seq.shape[1]
    peds = x_seq.shape[0]
    x_seq_velocity = np.zeros_like(x_seq)
    episa = 1e-6
    for i in range(1, length):
        for j in range(peds):
            position = x_seq[j][i]
            before_position = x_seq[j][i-1]
            position_norm = np.linalg.norm(position)
            before_position_norm = np.linalg.norm(before_position)
            if position_norm < episa:
                velocity = np.array([0,0])
            else:
                if before_position_norm < episa:
                    velocity = np.array([0, 0])
                else:
                    velocity = (position - before_position)/0.4
            x_seq_velocity[j][i] = velocity
    return x_seq_velocity

def translation(x_seq):
    first_frame = x_seq[:, 0, :]
    first_frame_new = first_frame[:, np.newaxis, :] #peds*1*2
    x_seq_translated = x_seq - first_frame_new
    return x_seq_translated

def augment_data(data):
    ks = [1, 2, 3]
    data_ = copy.deepcopy(data)  # data without rotation, used so rotated data can be appended to original df
    #k2rot = {1: '_rot90', 2: '_rot180', 3: '_rot270'}
    for k in ks:
        for t in range(len(data)):
            data_rot = rot(data[t], k)
            data_.append(data_rot)
    for t in range(27*4):
        data_flip = fliplr(data_[t])

        data_.append(data_flip)

    return data_

def rot(data_traj, k=1):
    xy = data_traj.copy()

    c, s = np.cos(-k * np.pi / 2), np.sin(-k * np.pi / 2)
    R = np.array([[c, s], [-s, c]])
    for i in range(20):
        xy[:, i, :] = np.dot(xy[:, i, :], R)

    return xy

def fliplr(data_traj):
    xy = data_traj.copy()

    R = np.array([[-1, 0], [0, 1]])
    for i in range(20):
        xy[:, i, :] = np.dot(xy[:, i, :], R)

    return xy

def calculate_loss(criterion, future, predictions, invtau_and_distribution, prior):

    invtau, miu_invtau, sigma_invtau = invtau_and_distribution[:, :, 0], invtau_and_distribution[:, :, 1], invtau_and_distribution[:, :, 2]
    prior_miu, prior_sigma = prior[0,:], prior[1,:]
    lqw =  torch.sum(- (invtau - miu_invtau)**2 / (2 * sigma_invtau**2) - torch.log(sigma_invtau) - float(0.5 * np.log(2*np.pi)))
    lpw =  torch.sum(- (torch.transpose(invtau, 0, 1) - prior_miu)**2 / (2 * prior_sigma**2) - torch.log(prior_sigma) - float(0.5 * np.log(2*np.pi)))
    ADL_traj = criterion(future, predictions)  # -p(D/tau)
    loss = lqw - lpw + ADL_traj

    return loss

def calculate_loss_f1f2(criterion, future, predictions, coefficients_1D_and_distribution, prior_coefficients, k_env_distribution, prior_env):

    coefficients, miu_coefficients, sigma_coefficients = coefficients_1D_and_distribution[:,0], coefficients_1D_and_distribution[:,1], coefficients_1D_and_distribution[:,2]
    miu_priorcoefficients, sigma_priorcoefficients = prior_coefficients[0], prior_coefficients[1]
    k_env, miu_k_env, sigma_k_env = k_env_distribution[:,0], k_env_distribution[:,1], k_env_distribution[:,2]
    miu_priorkenv, sigma_priorkenv = prior_env[0], prior_env[1]
    lqw_coefficients = torch.sum(- (coefficients - miu_coefficients)**2 / (2 * sigma_coefficients**2) - torch.log(sigma_coefficients)  - float(0.5 * np.log(2*np.pi)))
    lpw_coefficients = torch.sum(- (coefficients - miu_priorcoefficients) ** 2 / (2 * sigma_priorcoefficients ** 2) - torch.log(sigma_priorcoefficients) - float(0.5 * np.log(2 * np.pi)))
    lqw_kenv = torch.sum(- (k_env - miu_k_env) ** 2 / (2 * sigma_k_env ** 2) - torch.log(sigma_k_env) - float(0.5 * np.log(2 * np.pi)))
    lpw_kenv = torch.sum(- (k_env - miu_priorkenv) ** 2 / (2 * sigma_priorkenv ** 2) - torch.log(sigma_priorkenv) - float(0.5 * np.log(2 * np.pi)))

    ADL_traj = criterion(future, predictions)

    loss =  (lqw_coefficients + lqw_kenv) - (lpw_coefficients + lpw_kenv) + ADL_traj
    return loss

def calculate_loss_f2(criterion, future, predictions, k_env_distribution, prior_env):

    k_env, miu_k_env, sigma_k_env = k_env_distribution[:,0], k_env_distribution[:,1], k_env_distribution[:,2]
    miu_priorkenv, sigma_priorkenv = prior_env[0], prior_env[1]
    lqw_kenv = torch.sum(- (k_env - miu_k_env) ** 2 / (2 * sigma_k_env ** 2) - torch.log(sigma_k_env) - float(0.5 * np.log(2 * np.pi)))
    lpw_kenv = torch.sum(- (k_env - miu_priorkenv) ** 2 / (2 * sigma_priorkenv ** 2) - torch.log(sigma_priorkenv) - float(0.5 * np.log(2 * np.pi)))

    ADL_traj = criterion(future, predictions)

    loss =   lqw_kenv - lpw_kenv + ADL_traj
    return loss

def calculate_loss_f1(criterion, future, predictions, coefficients_1D_and_distribution, prior_coefficients):

    coefficients, miu_coefficients, sigma_coefficients = coefficients_1D_and_distribution[:,0], coefficients_1D_and_distribution[:,1], coefficients_1D_and_distribution[:,2]
    miu_priorcoefficients, sigma_priorcoefficients = prior_coefficients[0], prior_coefficients[1]
    lqw_coefficients = torch.sum(- (coefficients - miu_coefficients)**2 / (2 * sigma_coefficients**2) - torch.log(sigma_coefficients)  - float(0.5 * np.log(2*np.pi)))
    lpw_coefficients = torch.sum(- (coefficients - miu_priorcoefficients) ** 2 / (2 * sigma_priorcoefficients ** 2) - torch.log(sigma_priorcoefficients) - float(0.5 * np.log(2 * np.pi)))

    ADL_traj = criterion(future, predictions)

    loss =  lqw_coefficients  - lpw_coefficients  + ADL_traj
    return loss

def calculate_loss_cvae(mean, log_var, criterion, future, predictions):
    # reconstruction loss
    ADL_traj = criterion(future, predictions) # better with l2 loss

    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return KLD, ADL_traj
def translation_goals(goals, x_seq):
    first_frame = x_seq[:, 0, :]
    goals_translated = goals - first_frame #peds*2
    return goals_translated

def select_para(model_complete):

    params_totrain = []
    params_totrain.extend(model_complete.cell2.parameters())
    params_totrain.extend(model_complete.input_embedding_layer2.parameters())
    params_totrain.extend(model_complete.output_layer2.parameters())
    params_totrain.extend(model_complete.encoder_people_state.parameters())
    params_totrain.extend(model_complete.dec_para_people.parameters())
    return params_totrain

def new_point(checkpoint_t_dic, checkpoint_i_dic):
    point = checkpoint_i_dic
    dk_t = list(checkpoint_t_dic.keys())
    dk_i = list(point.keys())
    for k in range(57):
        point[dk_i[k]] = checkpoint_t_dic[dk_t[k]]
    return point

def translation_supp(supplemnt, x_seq):
    first_frame = x_seq[:, 0, :]
    for ped in range(supplemnt.shape[0]):
        for frame in range(20):
            all_other_peds = int(supplemnt[ped, frame, -1, 1])
            supplemnt[ped, frame, :all_other_peds, :2] = supplemnt[ped, frame, :all_other_peds, :2] - first_frame[ped,:]
    return supplemnt

def setup_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False