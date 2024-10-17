import torch
import torch.nn as nn
import numpy as np

def stateutils_desired_directions(current_step, generated_dest):

    destination_vectors = generated_dest - current_step #peds*2
    norm_factors = torch.norm(destination_vectors, dim=-1) #peds
    norm_factors = torch.unsqueeze(norm_factors, dim=-1)
    directions = destination_vectors / (norm_factors + 1e-8) #peds*2
    return directions

def f_ab_fun(current_step, coefficients, current_supplement, device):
    disp_p_x = torch.zeros(1,2)
    disp_p_y = torch.zeros(1,2)
    disp_p_x[0, 0] = 0.1
    disp_p_y[0, 1] = 0.1

    c1 = current_supplement[:,:-1,:2] #peds*maxpeds*2
    pedestrians = torch.unsqueeze(current_step, dim=1)  # peds*1*2

    v = value_p_p(c1, pedestrians, coefficients) # peds

    delta = torch.tensor(1e-3).to(device)
    dx = torch.tensor([[[delta, 0.0]]]).to(device) #1*1*2
    dy = torch.tensor([[[0.0, delta]]]).to(device) #1*1*2

    dvdx = (value_p_p(c1 + dx, pedestrians, coefficients) - v) / delta # peds
    dvdy = (value_p_p(c1 + dy, pedestrians, coefficients) - v) / delta # peds

    grad_r_ab = torch.stack((dvdx, dvdy), dim=-1) # peds*2
    out = -1.0 * grad_r_ab

    return out

def value_p_p(c1, pedestrians, coefficients):
    #potential field function : pf = K*exp(-norm(p-p1))

    d_p_c1 = pedestrians - c1  # peds*maxpeds*2
    d_p_c1_norm = torch.norm(d_p_c1, dim=-1) # peds*maxpeds

    potential = coefficients * torch.exp(-d_p_c1_norm.detach()) #peds*maxpeds

    out = torch.sum(potential, 1) #peds

    return out


'''MLP model'''
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            self.layers.append(nn.BatchNorm1d(dims[i+1]))
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(int(len(self.layers)/2)):
            x = self.layers[2*i](x)

            if i != int(len(self.layers)/2)-1:
                x = self.layers[2 * i + 1](x)
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x

class BNSP_attraction(nn.Module):

    def __init__(self, input_size, embedding_size, rnn_size, output_size, encoder_dest_size, dec_tau_size):
        '''
        Args:
            input_size: Dimension of the state
            embedding_size: Dimension of the input of LSTM
            rnn_size: Dimension of LSTM
            output_size: Dimension of linear transformation of output of LSTM
            encoder_dest_size: Hitten size of MLP encoding destinations
            dec_tau_size: Hitten size of MLP decoding extracted features to tau

        '''
        super(BNSP_attraction, self).__init__()

        # The Goal-Network
        self.cell = nn.LSTMCell(embedding_size, rnn_size)
        self.input_embedding_layer = nn.Linear(input_size, embedding_size)
        self.output_layer = nn.Linear(rnn_size, output_size)

        self.encoder_dest = MLP(input_dim = 2, output_dim = output_size, hidden_size=encoder_dest_size)
        self.dec_tau = MLP(input_dim = 2*output_size, output_dim = 2, hidden_size=dec_tau_size)


        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.sample_times = 3


    def forward_lstm(self, input_lstm, hidden_states_current, cell_states_current):
        #input_lstm: peds*4

        # Embed inputs
        input_embedded = self.dropout(self.relu(self.input_embedding_layer(input_lstm))) #peds*embedding_size
        h_nodes, c_nodes = self.cell(input_embedded, (hidden_states_current, cell_states_current)) #h_nodes/c_nodes: peds*rnn_size
        outputs = self.output_layer(h_nodes) #peds*output_size

        return outputs, h_nodes, c_nodes

    def forward_next_step(self, current_step, current_vel, initial_speeds, dest, features_lstm, infer = False, device=torch.device('cpu')):

        delta_t = torch.tensor(0.4)
        e = stateutils_desired_directions(current_step, dest)  #peds*2

        features_dest = self.encoder_dest(dest)
        features_tau = torch.cat((features_lstm, features_dest), dim = -1)
        distribution_tau = self.dec_tau(features_tau)
        miu_invtau, logsigma_invtau = distribution_tau[:, 0] , distribution_tau[:, 1]
        if infer:
            F0 = torch.unsqueeze(miu_invtau, dim=1) * (initial_speeds * e - current_vel)  # peds*2

            F = F0  # peds*2

            w_v = current_vel + delta_t * F  # peds*2

            # update state
            prediction = current_step + w_v * delta_t  # peds*2

            return prediction, w_v


        else:
            invtau = torch.zeros_like(miu_invtau)
            sigma_invtau = torch.exp(logsigma_invtau)
            for i in range(self.sample_times):
                epsilon = torch.Tensor(len(miu_invtau)).normal_(0, 1).to(device)
                invtau += (miu_invtau + sigma_invtau * epsilon) / self.sample_times
                #tau += (miu_tau + sigma_tau * epsilon) / self.sample_times
            invtau_and_distribution = torch.stack([invtau, miu_invtau, sigma_invtau], dim=1)
            F0 = torch.unsqueeze(invtau, dim=1) * (initial_speeds * e - current_vel)  #peds*2

            F = F0 #peds*2

            w_v = current_vel + delta_t * F  #peds*2

            # update state
            prediction = current_step + w_v * delta_t  # peds*2


            return prediction, w_v, invtau_and_distribution

    def calculate_F0(self, current_step, current_vel, initial_speeds, dest, features_lstm, device=torch.device('cpu')):


        e = stateutils_desired_directions(current_step, dest)  #peds*2

        features_dest = self.encoder_dest(dest)
        features_tau = torch.cat((features_lstm, features_dest), dim = -1)
        distribution_tau = self.dec_tau(features_tau)
        miu_invtau, logsigma_invtau = distribution_tau[:, 0] , distribution_tau[:, 1:2]

        F0 = torch.unsqueeze(miu_invtau, dim=1) * (initial_speeds * e - current_vel)  # peds*2
        F0_std_part = torch.abs(initial_speeds * e - current_vel)
        sigma_invtau = torch.exp(logsigma_invtau)


        return F0, sigma_invtau * F0_std_part

    def sample_F0(self, current_step, current_vel, initial_speeds, dest, features_lstm, device=torch.device('cpu')):

        e = stateutils_desired_directions(current_step, dest)  #peds*2

        features_dest = self.encoder_dest(dest)
        features_tau = torch.cat((features_lstm, features_dest), dim = -1)
        distribution_tau = self.dec_tau(features_tau)
        miu_invtau, logsigma_invtau = distribution_tau[:, 0] , distribution_tau[:, 1]
        sigma_invtau = 20*torch.exp(logsigma_invtau)

        F0_samples = []
        F0 = torch.unsqueeze(miu_invtau, dim=1) * (initial_speeds * e - current_vel)  # peds*2
        F0_samples.append(F0)
        for i in range(2):
            epsilon = torch.Tensor(len(miu_invtau)).normal_(0, 1).to(device)
            invtau = miu_invtau + sigma_invtau * epsilon
            F0 = torch.unsqueeze(invtau, dim=1) * (initial_speeds * e - current_vel)  # peds*2
            F0_samples.append(F0)



        return F0_samples


