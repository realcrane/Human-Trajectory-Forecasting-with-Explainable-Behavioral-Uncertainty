from model_repulsion import *
from models_goals import *
from model_cvae import *
from torch.autograd import Variable
from utils import *
import torch.optim as optim
import argparse
import os
import pickle
import cv2
import yaml
def train(path, scenes):

    model_cvae.train()
    model_attraction.eval()
    model_repulsion.eval()
    train_loss = 0
    total_kld, total_adl = 0, 0
    criterion = nn.MSELoss()

    shuffle_index = torch.randperm(30)

    for t in shuffle_index:
        scene = scenes[t]
        load_name = path + scene
        with open(load_name, 'rb') as f:
            data = pickle.load(f)
        traj_complete, supplement, first_part = data[0], data[1], data[2]
        traj_complete = np.array(traj_complete)
        if len(traj_complete.shape) == 1:
            continue
        first_frame = traj_complete[:, 0, :2]
        traj_translated = translation(traj_complete[:, :, :2])
        traj_complete_translated = np.concatenate((traj_translated, traj_complete[:, :, 2:]), axis=-1)
        supplement_translated = translation_supp(supplement, traj_complete[:, :, :2])

        traj, supplement = torch.DoubleTensor(traj_complete_translated).to(device), torch.DoubleTensor(
            supplement_translated).to(device)
        first_frame = torch.DoubleTensor(first_frame).to(device)

        semantic_map = cv2.imread(semantic_path_train + semantic_maps_name_train[t])
        semantic_map = np.transpose(semantic_map[:, :, 0])

        y = traj[:, params['past_length']:, :2]  # peds*future_length*2
        dest = y[:, -1, :].to(device)
        future = y.contiguous().to(device)

        future_vel = (dest - traj[:, params['past_length'] - 1, :2]) / (torch.tensor(params['future_length']).to(device) * 0.4) #peds*2
        future_vel_norm = torch.norm(future_vel, dim=-1) #peds
        initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1) #peds*1

        num_peds = traj.shape[0]
        numNodes = num_peds

        hidden_states1 = Variable(torch.zeros(numNodes, params['rnn_size']))
        cell_states1 = Variable(torch.zeros(numNodes, params['rnn_size']))
        hidden_states1 = hidden_states1.to(device)
        cell_states1 = cell_states1.to(device)
        hidden_states2 = Variable(torch.zeros(numNodes, params['rnn_size']))
        cell_states2 = Variable(torch.zeros(numNodes, params['rnn_size']))
        hidden_states2 = hidden_states2.to(device)
        cell_states2 = cell_states2.to(device)

        for m in range(1, params['past_length']):  #
            current_step = traj[:, m, :2]  # peds*2
            current_vel = traj[:, m, 2:]  # peds*2
            input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds*4
            with torch.no_grad():
                outputs_features1, hidden_states1, cell_states1 = model_attraction.forward_lstm(input_lstm, hidden_states1, cell_states1)
                outputs_features2, hidden_states2, cell_states2 = model_repulsion.forward_lstm(input_lstm, hidden_states2, cell_states2)
        with torch.no_grad():
            coefficients, curr_supp, _ = model_repulsion.forward_coefficient_people(outputs_features2, supplement[:, 7, :, :], current_step, current_vel, device, infer=True)  # peds*maxpeds*2, peds*(max_peds + 1)*4
            F_goal = model_attraction.calculate_F0(current_step, current_vel, initial_speeds, dest, outputs_features1, device=device)
            F_env = model_repulsion.calculate_F2(current_step, current_vel, semantic_map, first_frame, miu_k_env, sigma_k_env, F_goal, infer=True, device=device)
            prediction, w_v = model_repulsion.forward_next_step(current_step, current_vel, coefficients, curr_supp,
                                                                sigma, F_goal, F_env, device=device)
        x = torch.zeros((num_peds, 9, 2)).to(device)
        x[:, :8, :] = copy.deepcopy(traj[:, :8, :2])
        x[:, -1, :] = prediction
        x = torch.reshape(x, (-1, x.shape[1] * x.shape[2])).to(device) * params['data_scale']
        alpha = (traj[:, 8, :2] - prediction)*params['data_scale']
        alpha_recon, mu, var = model_cvae.forward(x, next_step=alpha, device=device)
        optimizer.zero_grad()
        kld, adl = calculate_loss_cvae(mu, var, criterion, alpha, alpha_recon)
        loss = kld * params["kld_reg"] + adl
        loss.backward()

        train_loss += loss.item()
        total_kld += kld.item()
        total_adl += adl.item()
        optimizer.step()

        for i in range(1, params['future_length']):
            current_step = traj[:, 7+i, :2]  # peds*2
            current_vel = traj[:, 7+i, 2:]  # peds*2
            input_lstm = torch.cat((current_step, current_vel), dim=1)
            with torch.no_grad():
                outputs_features1, hidden_states1, cell_states1 = model_attraction.forward_lstm(input_lstm, hidden_states1, cell_states1)
                outputs_features2, hidden_states2, cell_states2 = model_repulsion.forward_lstm(input_lstm, hidden_states2, cell_states2)

            future_vel = (dest - traj[:, 7+i, :2]) / ((12-i) * 0.4)  # peds*2
            future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
            initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1
            with torch.no_grad():
                coefficients, curr_supp, _ = model_repulsion.forward_coefficient_people(outputs_features2, supplement[:, 7+i, :, :],  current_step, current_vel, device, infer=True)  # peds*maxpeds*2, peds*(max_peds + 1)*4
                F_goal = model_attraction.calculate_F0(current_step, current_vel, initial_speeds, dest, outputs_features1, device=device)
                F_env = model_repulsion.calculate_F2(current_step, current_vel, semantic_map, first_frame, miu_k_env, sigma_k_env, F_goal, infer=True, device=device)
                prediction, w_v = model_repulsion.forward_next_step(current_step, current_vel, coefficients, curr_supp,
                                                                    sigma, F_goal, F_env, device=device)
            x = torch.zeros((num_peds, 9, 2)).to(device)
            x[:, :8, :] = copy.deepcopy(traj[:, i : 8 + i, :2])
            x[:, -1, :] = prediction
            first_frame_x = copy.deepcopy(x[:, :1, :])
            x = x - first_frame_x
            x = torch.reshape(x,(-1, x.shape[1] * x.shape[2])).to(device) * params['data_scale']
            alpha = (traj[:, 8+i, :2] - prediction) * params['data_scale']
            alpha_recon, mu, var = model_cvae.forward(x, next_step=alpha, device=device)


            optimizer.zero_grad()
            kld, adl = calculate_loss_cvae(mu, var, criterion, alpha, alpha_recon)
            loss = kld * params["kld_reg"] + adl
            loss.backward()

            train_loss += loss.item()
            total_kld += kld.item()
            total_adl += adl.item()
            optimizer.step()

    return train_loss, total_kld, total_adl

def test(path, scenes, generated_goals, best_of_n):
    model_attraction.eval()
    model_repulsion.eval()
    model_cvae.eval()
    all_ade = []
    all_fde = []
    index = 0
    all_traj = []
    all_scenes = []

    with torch.no_grad():
        #for i, scene in enumerate(scenes):
        for i in range(5, len(scenes)):
            scene = scenes[i]
            load_name = path + scene
            with open(load_name, 'rb') as f:
                data = pickle.load(f)
            traj_complete, supplement, first_part = data[0], data[1], data[2]
            traj_complete = np.array(traj_complete)
            if len(traj_complete.shape) == 1:
                index += 1
                continue
            traj_translated = translation(traj_complete[:, :, :2])
            traj_complete_translated = np.concatenate((traj_translated, traj_complete[:, :, 2:]), axis=-1)
            supplement_translated = translation_supp(supplement, traj_complete[:, :, :2])
            traj, supplement = torch.DoubleTensor(traj_complete_translated).to(device), torch.DoubleTensor(
                supplement_translated).to(device)
            traj_copy = copy.deepcopy(traj)

            semantic_map = cv2.imread(semantic_path_test + semantic_maps_name_test[i])
            semantic_map = np.transpose(semantic_map[:, :, 0])
            y = traj[:, params['past_length']:, :2]  # peds*future_length*2
            y = y.cpu().numpy()
            first_frame = torch.DoubleTensor(traj_complete[:, 0, :2]).to(device)  # peds*2
            num_peds = traj.shape[0]
            ade_20 = np.zeros((20, len(traj_complete)))
            fde_20 = np.zeros((20, len(traj_complete)))
            predictions_20 = np.zeros((20, num_peds, params['future_length'], 2))

            for j in range(20):
                goals_translated = translation_goals(generated_goals[1][i - index][j, :, :], traj_complete[:, :, :2])  # 20*peds*2
                dest = torch.DoubleTensor(goals_translated).to(device)

                #dest = traj[:,-1,:2]
                future_vel = (dest - traj[:, params['past_length'] - 1, :2]) / (torch.tensor(params['future_length']).to(device) * 0.4)  # peds*2
                future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
                initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1

                numNodes = num_peds

                hidden_states1 = Variable(torch.zeros(numNodes, params['rnn_size']))
                cell_states1 = Variable(torch.zeros(numNodes, params['rnn_size']))
                hidden_states1 = hidden_states1.to(device)
                cell_states1 = cell_states1.to(device)
                hidden_states2 = Variable(torch.zeros(numNodes, params['rnn_size']))
                cell_states2 = Variable(torch.zeros(numNodes, params['rnn_size']))
                hidden_states2 = hidden_states2.to(device)
                cell_states2 = cell_states2.to(device)

                for m in range(1, params['past_length']):  #
                    current_step = traj[:, m, :2]  # peds*2
                    current_vel = traj[:, m, 2:]  # peds*2
                    input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds*4
                    outputs_features1, hidden_states1, cell_states1 = model_attraction.forward_lstm(input_lstm, hidden_states1, cell_states1)
                    outputs_features2, hidden_states2, cell_states2 = model_repulsion.forward_lstm(input_lstm, hidden_states2, cell_states2)
                predictions = torch.zeros(num_peds, params['future_length'], 2).to(device)
                prediction_forcesamples_cvae = torch.zeros((best_of_n, num_peds, 2)).to(device)

                F_goal_samples = model_attraction.sample_F0(current_step, current_vel, initial_speeds, dest, outputs_features1, device=device)
                coefficients_samples, curr_supp = model_repulsion.sample_forward_coefficient_people(outputs_features2,
                                                                                        supplement[:, 7, :, :],
                                                                                        current_step, current_vel, F_goal_samples, device)
                F_env = model_repulsion.calculate_F2(current_step, current_vel, semantic_map, first_frame, miu_k_env, sigma_k_env, F_goal_samples[0], infer=True, device=device)
                prediction_forcesamples, w_v_forcesamples = model_repulsion.sample_forward_next_step(current_step, current_vel, coefficients_samples, curr_supp, sigma, F_goal_samples, F_env, device=device)
                N_force_samples = len(prediction_forcesamples)
                N_cvae_samples = int(best_of_n/N_force_samples)
                for n in range(N_force_samples):
                    x = torch.zeros((num_peds, 9, 2)).to(device)
                    x[:,:8,:] = traj_copy[:, :8, :2]
                    x[:,-1,:] = prediction_forcesamples[n]
                    x = torch.reshape(x, (-1, x.shape[1] * x.shape[2])).to(device) * params['data_scale']
                    alpha_step = torch.zeros(N_cvae_samples, len(traj), 2).to(device)
                    for s in range(N_cvae_samples):
                        alpha_recon = model_cvae.forward(x, device=device)
                        alpha_step[s, :, :] = alpha_recon
                    alpha_step[-1, :, :] = torch.zeros_like(alpha_step[-1, :, :])
                    prediction_correct = alpha_step / params['data_scale'] + prediction_forcesamples[n]
                    prediction_forcesamples_cvae[n*N_cvae_samples:(n+1)*N_cvae_samples,:,:] = prediction_correct
                predictions_norm = torch.norm((prediction_forcesamples_cvae - traj[:, 8, :2]), dim=-1)
                values, indices = torch.min(predictions_norm, dim=0)  # peds
                ns_recon_best = prediction_forcesamples_cvae[indices, [x for x in range(len(traj))], :]  # peds*2
                predictions[:, 0, :] = ns_recon_best
                current_step = ns_recon_best
                current_vel = (ns_recon_best - traj_copy[:, 7, :2]) / 0.4
                traj_copy[:, 8, :2] = current_step
                traj_copy[:, 8, 2:] = current_vel

                for t in range(params['future_length'] - 1):
                    input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds*4
                    outputs_features1, hidden_states1, cell_states1 = model_attraction.forward_lstm(input_lstm, hidden_states1, cell_states1)
                    outputs_features2, hidden_states2, cell_states2 = model_repulsion.forward_lstm(input_lstm, hidden_states2, cell_states2)

                    future_vel = (dest - current_step) / ((12 - t - 1) * 0.4)  # peds*2
                    future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
                    initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1

                    F_goal_samples = model_attraction.sample_F0(current_step, current_vel, initial_speeds, dest,
                                                           outputs_features1, device=device)
                    coefficients_samples, curr_supp = model_repulsion.sample_forward_coefficient_test(outputs_features2, supplement[:, 7 + t + 1, :, :],
                                                                     current_step, current_vel, first_part, first_frame, F_goal_samples,
                                                                     device=device)
                    F_env = model_repulsion.calculate_F2(current_step, current_vel, semantic_map, first_frame,
                                                          miu_k_env, sigma_k_env, F_goal_samples[0], infer=True, device=device)
                    prediction_forcesamples, w_v_forcesamples = model_repulsion.sample_forward_next_step(current_step, current_vel,
                                                                                        coefficients_samples, curr_supp, sigma,
                                                                                        F_goal_samples, F_env, device=device)
                    for n in range(N_force_samples):
                        x = torch.zeros((num_peds, 9, 2)).to(device)
                        x[:, :8, :] = traj_copy[:, t+1 :8 + t+1, :2]
                        x[:, -1, :] = prediction_forcesamples[n]
                        x = torch.reshape(x, (-1, x.shape[1] * x.shape[2])).to(device) * params['data_scale']
                        alpha_step = torch.zeros(N_cvae_samples, len(traj), 2).to(device)
                        for s in range(N_cvae_samples):
                            alpha_recon = model_cvae.forward(x, device=device)
                            alpha_step[s, :, :] = alpha_recon
                        alpha_step[-1, :, :] = torch.zeros_like(alpha_step[-1, :, :])
                        prediction_correct = alpha_step / params['data_scale'] + prediction_forcesamples[n]
                        prediction_forcesamples_cvae[n * N_cvae_samples:(n + 1) * N_cvae_samples, :, :] = prediction_correct
                    predictions_norm = torch.norm((prediction_forcesamples_cvae - traj[:, 8+t+1, :2]), dim=-1)
                    values, indices = torch.min(predictions_norm, dim=0)  # peds
                    ns_recon_best = prediction_forcesamples_cvae[indices, [x for x in range(len(traj))], :]  # peds*2
                    predictions[:, t+1, :] = ns_recon_best
                    current_step = ns_recon_best
                    current_vel = (ns_recon_best - traj_copy[:, 7+t+1, :2]) / 0.4
                    traj_copy[:, 8+t+1, :2] = current_step
                    traj_copy[:, 8+t+1, 2:] = current_vel
                predictions_20[j] = predictions + torch.unsqueeze(first_frame, dim=1)
                predictions = predictions.cpu().numpy()

                # ADE error
                test_ade = np.mean(np.linalg.norm(y - predictions, axis = 2), axis=1) # peds
                test_fde = np.linalg.norm((y[:,-1,:] - predictions[:, -1, :]), axis=1) #peds
                ade_20[j, :] = test_ade
                fde_20[j, :] = test_fde
            ade_single = np.min(ade_20, axis=0)  # peds
            fde_single = np.min(fde_20, axis=0)  # peds
            all_ade.append(ade_single)
            all_fde.append(fde_single)
            all_traj.append(predictions_20)
            all_scenes.append(scene)
            save_list = [predictions_20, ade_20, fde_20]
            with open('failure_cases/data/pred_' + scene + '_for_failure.pkl', 'wb') as f:
                pickle.dump(save_list, f)

            #print('test finish:', i)
        ade = np.mean(np.concatenate(all_ade))
        fde = np.mean(np.concatenate(all_fde))
    return ade, fde

parser = argparse.ArgumentParser(description='NSP')

parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--save_file', '-sf', type=str, default='CVAE_1.52.pt')
parser.add_argument('--verbose', '-v', action='store_true')

args = parser.parse_args()

CONFIG_FILE_PATH = 'config/sdd_bnsp.yaml'  # yaml config file containing all the hyperparameters
with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
print(params)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
print(device)

model_attraction = BNSP_attraction(params["input_size"], params["embedding_size"], params["rnn_size"], params["output_size"],  params["enc_size"], params["dec_size"])
model_attraction = model_attraction.double().to(device)
model_repulsion = BNSP_repulsion(params["input_size"], params["embedding_size"], params["rnn_size"], params["output_size"],  params["enc_size"], params["dec_size"])
model_repulsion = model_repulsion.double().to(device)
model_cvae = CVAE(params["enc_past_size_cvae"], params["enc_dest_size_cvae"], params["enc_latent_size_cvae"], params["dec_size_cvae"], params["fdim"], params["zdim"], params["sigma"], params["past_length"], params["future_length"], args.verbose)
model_cvae = model_cvae.double().to(device)

load_path_f0 = 'saved_models/SDD_goals.pt'
checkpoint_f0 = torch.load(load_path_f0, map_location=device)
model_attraction.load_state_dict(checkpoint_f0['model_state_dict'])
load_path_f1 = 'saved_models/SDD_repulsion.pt'
checkpoint_f1 = torch.load(load_path_f1, map_location=device)
model_repulsion.load_state_dict(checkpoint_f1['model_state_dict'])
# load_path_cvae = 'saved_models/CVAE_1.52.pt'
# checkpoint_cvae = torch.load(load_path_cvae, map_location=device)
# model_cvae.load_state_dict(checkpoint_cvae['model_state_dict'])


sigma = torch.tensor(100)
miu_k_env = torch.tensor(65.0).to(device)
sigma_k_env = torch.tensor(10.0).to(device)

optimizer = optim.Adam(model_cvae.parameters(), lr=  params["learning_rate"])

best_test_loss = 1.78 # start saving after this threshold
best_endpoint_loss = 3.44
goals_path = 'data/SDD/goals_Ynet.pickle'
with open(goals_path, 'rb') as f:
    goals = pickle.load(f)


path_train = 'data/SDD/train_pickle/'
scenes_train = os.listdir(path_train)
path_test = 'data/SDD/test_pickle/'
scenes_test = os.listdir(path_test)
semantic_path_train = 'data/SDD/train_masks/'
semantic_maps_name_train = os.listdir(semantic_path_train)
semantic_path_test = 'data/SDD/test_masks/'
semantic_maps_name_test = os.listdir(semantic_path_test)
N_stepsample = 15
for e in range(params['num_epochs']):
    train_loss, kld, adl = train(path_train, scenes_train)
    test_ade, test_fde = test(path_test, scenes_test, goals, N_stepsample)
    print('test_ade, test_fde:', test_ade, test_fde)

    print()

    if test_ade < best_test_loss:
        best_test_loss = test_ade
        best_endpoint_loss = test_fde
        print("Epoch: ", e)
        print('################## BEST PERFORMANCE {:0.2f} ########'.format(test_ade))
        save_path = 'saved_models/' + args.save_file
        torch.save({
            'hyper_params': params,
            'model_state_dict': model_cvae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, save_path)
        print("Saved model to:\n{}".format(save_path))

    print('num_epoch', e)
    print("Train Loss", train_loss)
    print("KLD", kld)
    print("ADL", adl)
    print('Current Test ADE', test_ade)
    print('Current Test FDE', test_fde)
    print("Test Best ADE Loss So Far (N = {})".format(N_stepsample), best_test_loss)
    print("Test Best FDE Loss So Far (N = {})".format(N_stepsample), best_endpoint_loss)