"""# **Class: KalmanNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func
from visual_supplementary import y_size, decoaded_dimention


class KalmanNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #############
    ### Build ###
    #############
    def Build(self, ssModel, h_fully_connected,model_AE_trained):

        self.InitSystemDynamics(ssModel.F, h_fully_connected) #entering new H

        # Number of neurons in the 1st hidden layer
        H1_KNet = ssModel.m + ssModel.d
        #(ssModel.m + ssModel.n)

        # Number of neurons in the 2nd hidden layer
        H2_KNet = (ssModel.m * ssModel.n)

        self.InitKGainNet(H1_KNet, H2_KNet)
        self.model_AE = model_AE_trained

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, H1, H2):
        # Input Dimensions
        D_in = self.m + self.d  # x(t-1), y(t)
        # self.m + self.n
        # Output Dimensions
        D_out = self.m * self.d  # Kalman Gain
        # self.m * self.n
        ###################
        ### Input Layer ###
        ###################
        # Linear Layer
        self.KG_l1 = torch.nn.Linear(D_in, H1, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu1 = torch.nn.ReLU()

        ###########
        ### GRU ###
        ###########
        # Input Dimension
        self.input_dim = H1
        # Hidden Dimension
        self.hidden_dim = self.d + self.n
        #self.m + self.n
        # Number of Layers10
        self.n_layers = 1
        # Batch Size
        self.batch_size = 1
        # Input Sequence Length
        self.seq_len_input = 1
        # Hidden Sequence Length
        self.seq_len_hidden = self.n_layers

        # batch_first = False
        # dropout = 0.1 ;

        # Initialize a Tensor for GRU Input
        # self.GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim)

        # Initialize a Tensor for Hidden State
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim).to(self.device,non_blocking = True)

        # Iniatialize GRU Layer
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers)#take time

        ####################
        ### Hidden Layer ###
        ####################
        self.KG_l2 = torch.nn.Linear(self.hidden_dim, H2, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu2 = torch.nn.ReLU()

        ####################
        ### Output Layer ###
        ####################
        self.KG_l3 = torch.nn.Linear(H2, D_out, bias=True)

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, F, H):
        # Set State Evolution Matrix
        self.F = F.to(self.device,non_blocking = True)
        self.F_T = torch.transpose(F, 0, 1)
        self.m = self.F.size()[0]
        self.d = decoaded_dimention

        # Set Observation Matrix
        self.H_FC = H
        #H.to(self.device,non_blocking = True)
        #self.H_T = torch.transpose(H, 0, 1)
        self.n = y_size*y_size

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0):

        self.m1x_prior = M1_0.to(self.device,non_blocking = True)

        self.m1x_posterior = M1_0.to(self.device,non_blocking = True)

        self.state_process_posterior_0 = M1_0.to(self.device,non_blocking = True)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):

        # Compute the 1-st moment of x based on model knowledge and without process noise
        self.state_process_prior_0 = torch.matmul(self.F, self.state_process_posterior_0)

        # Compute the 1-st moment of y based on model knowledge and without noise
        # H is learned FC
        self.H_FC.eval()
        self.obs_process_0 = self.H_FC(self.state_process_prior_0.reshape(1, self.m))
        # H is known matrix
        #self.obs_process_0 = torch.matmul(self.H_vis, self.state_process_prior_0)

        # Predict the 1-st moment of x
        self.m1x_prev_prior = self.m1x_prior
        self.m1x_prior = torch.matmul(self.F, self.m1x_posterior)

        # Predict the 1-st moment of y
        # H is learned FC
        self.H_FC.eval()
        self.m1y = self.H_FC(self.m1x_prior.reshape(1, self.m)).reshape(self.d, 1)
        # H is known matrix
        #self.m1y = torch.matmul(self.H_vis, self.m1x_prior)

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):

        # Reshape and Normalize the difference in X prior
        # Featture 4: x_t|t - x_t|t-1
        #dm1x = self.m1x_prior - self.state_process_prior_0
        dm1x = self.m1x_posterior - self.m1x_prev_prior
        dm1x_reshape = torch.squeeze(dm1x)
        dm1x_norm = func.normalize(dm1x_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Feature 2: yt - y_t+1|t
        dm1y = y - torch.squeeze(self.m1y)
        dm1y_norm = func.normalize(dm1y, p=2, dim=0, eps=1e-12, out=None)

        # KGain Net Input
        KGainNet_in = torch.cat([dm1y_norm, dm1x_norm], dim=0)

        # Kalman Gain Network Step
        KG = self.KGain_step(KGainNet_in)

        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.m, self.d))
        # (self.m, self.n)

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):
        # Compute Priors
        self.step_prior()
    # h=fhfhfsdhdh function
        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Innovation
        y_obs = torch.unsqueeze(y, 1)
        dy = y_obs - self.m1y

        # Compute the 1-st posterior moment
        INOV = torch.matmul(self.KGain, dy)
        self.m1x_posterior = self.m1x_prior + INOV

        # return
        return torch.squeeze(self.m1x_posterior)

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, KGainNet_in):

        ###################
        ### Input Layer ###
        ###################
        L1_out = self.KG_l1(KGainNet_in);
        La1_out = self.KG_relu1(L1_out);

        ###########
        ### GRU ###
        ###########
        GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim).to(self.device,non_blocking = True)
        GRU_in[0, 0, :] = La1_out
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn)
        GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim))

        ####################
        ### Hidden Layer ###
        ####################
        L2_out = self.KG_l2(GRU_out_reshape)
        La2_out = self.KG_relu2(L2_out)

        ####################
        ### Output Layer ###
        ####################
        L3_out = self.KG_l3(La2_out)
        return L3_out

    ###############
    ### Forward ###
    ###############
    def forward(self, yt):
        yt = yt.to(self.device,non_blocking = True)
        AE_input = yt.reshape((1, len(yt)))
        y_cv_decoaded_t = self.model_AE(AE_input)[0].squeeze()
        return self.KNet_step(y_cv_decoaded_t)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data