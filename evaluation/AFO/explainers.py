import random
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.mixture import GaussianMixture
import pandas as pd
import pickle

class GRUD_AFOExplainer:
    def __init__(self, model, data_means, timedelta_means, activation=torch.nn.Softmax(-1)):
        self.device = torch.device("cuda:2") 
        print(self.device)
        self.base_model = model.to(self.device)
        
        # data points is the 0th slice of the grud-loader
        self.data_distribution = data_means
        self.time_distribution = timedelta_means
        print('Initial dist shapes', self.data_distribution.shape, self.time_distribution.shape)
        self.activation = activation

    def attribute(self, x, x_padding, ind, retrospective=False):
        """
        Compute importance score for a sample x, over time and features
        :param x: Sample instance to evaluate score for. Shape:[batch, features, time]
        :param n_samples: number of Monte-Carlo samples
        :return: Importance score matrix of shape:[batch, features, time]
        """
        x = x.to(self.device) # set as batch, features, time in for loop below
        _, _, n_features, t_len = x.shape
        
        # model_x should be batch, time, features
        model_x = x.permute(0,1,3,2) ## CHANGE: dimensionality
        model_x = model_x.to(self.device)
        if retrospective:
            p_y_t = self.activation(self.base_model(model_x, x_padding)) ## change to be model_x

        ## CHANGE remove the loop of time here, since we are feeding in one timestep at a time instead of one batch of patients at a time
        if not retrospective:
            ## no longer need the :t+1 because I only have up to the given timestep in my model
            p_y_t = self.base_model(model_x, x_padding) ## change to be model_x; remove activation bc that's in my model
        
        score = torch.zeros((p_y_t.shape[0], n_features))
        for i in tqdm(range(n_features)):
            # self.data distribution should be x, feats, time and then we access all of the values for feature i to create the empirical distribution
            
            # FOR FEATURE
            long_feature_dist = (self.data_distribution[:, i, ind]).reshape(-1)
            feature_dist = long_feature_dist[long_feature_dist!=0].to(self.device)
            long_time_dist = (self.time_distribution[:, i, ind]).reshape(-1)
            time_dist = long_time_dist[long_feature_dist!=0].to(self.device)

            if len(feature_dist) == 0:
                print(i)
                score[:, i] = 0
            else:
                x_hat = x.clone() ##CHANGE to not depend on :t+1
                """
                kl_accum = []
                batch_size = x.shape[0]
                repeat_sizes = [5, 5]
                for repeat in repeat_sizes:
                    # Repeat x and padding for this chunk
                    x_hat_chunk = x.unsqueeze(0).repeat(repeat, 1, 1, 1, 1)  # [repeat, batch, 4, feat, time]
                    x_hat_chunk = x_hat_chunk.view(-1, 4, x.shape[2], x.shape[3])  # [repeat*batch, 4, feat, time]
                    
                    x_padding_chunk = x_padding.unsqueeze(0).repeat(repeat, 1, 1).view(-1, x_padding.shape[1])
                    
                    # Feature permutation for this chunk
                    rand_idx_feat = torch.randint(0, len(feature_dist), (repeat * batch_size,), device=self.device)
                    x_hat_chunk[:, 0, i, ind] = feature_dist[rand_idx_feat]
                    x_hat_chunk[:, 1, i, ind+1] = x_hat_chunk[:, 0, i, ind]
                    x_hat_chunk[:, 2, i, ind] = ((x_hat_chunk[:, 0, i, ind] - feature_dist.mean()) > 1e-3).to(x_hat_chunk.dtype)

                    # Time permutation for this chunk
                    rand_idx_time = torch.randint(0, len(time_dist), (repeat * batch_size,), device=self.device)
                    x_hat_chunk[:, 3, i, ind] = time_dist[rand_idx_time]

                    # run the model
                    print(x_hat_chunk.shape, x_padding_chunk.shape)
                    y_hat_flat = self.base_model(x_hat_chunk.permute(0,1,3,2), x_padding_chunk)  
                    y_hat_flat = y_hat_flat.view(repeat, -1)
                    print(y_hat_flat.shape)
                    # broadcast p to be this shape
                    print(p_y_t.shape)
                    p_y_t_exp = p_y_t.unsqueeze(0).expand_as(y_hat_flat)
                    print(p_y_t_exp.shape)
                    kl_chunk = torch.abs(y_hat_flat - p_y_t_exp).mean(dim=-1)  # [repeat]
                    print(kl_chunk.shape)
                    kl_accum.append(kl_chunk)
                kl_all = torch.cat(kl_accum, dim=0)
                print(kl_all.shape)
                E_kl = np.mean(np.array(kl_all), axis=0)
                print(E_kl.shape)
                score[:, i] = E_kl
                """
                kl_all=[]

                for _ in range(10):
                    x_hat = x.clone()
                    ## we want to permute time point IND
                    rand_idx = torch.randint(0, len(feature_dist), (len(x),), device=self.device)
                    x_hat[:, 0, i, ind] = feature_dist[rand_idx]
                    x_hat[:, 1, i, ind+1] = x_hat[:, 0, i, ind]
                    x_hat[:, 2, i, ind] = ((x_hat[:, 0, i, ind]-feature_dist.mean()) > 1e-3) * 1 # are we close to the mean? since we standard scaled, that means we are close to 0
                    rand_idx = torch.randint(0, len(time_dist), (len(x),), device=self.device)
                    x_hat[:, 3, i, ind] = time_dist[rand_idx]

                    y_hat_t = self.base_model(x_hat.permute(0,1,3,2), x_padding) ## CHANGE: permute
                    # kl = torch.nn.KLDivLoss(reduction='none')(torch.log(y_hat_t), p_y_t)
                    kl = torch.abs(y_hat_t - p_y_t)
                    # kl_all.append(torch.sum(kl, -1).cpu().detach().numpy())
                    kl_all.append(kl.detach())
                E_kl = torch.stack(kl_all).mean(dim=0)
                # score[:, i, t] = 2./(1+np.exp(-1*E_kl)) - 1.
                score[:, i] = E_kl.cpu()
                
        return score

class AFOExplainer:
    def __init__(self, model, data_means, activation=torch.nn.Softmax(-1)):
        self.device = torch.device("cuda:0") 
        print(self.device)
        self.base_model = model.to(self.device)
        
        # data points is the 0th slice of the grud-loader
        self.data_distribution = data_means
        print('Initial dist shapes', self.data_distribution.shape)
        self.activation = activation

    def attribute(self, x, x_padding, ind, retrospective=False):
        """
        Compute importance score for a sample x, over time and features
        :param x: Sample instance to evaluate score for. Shape:[batch, features, time]
        :param n_samples: number of Monte-Carlo samples
        :return: Importance score matrix of shape:[batch, features, time]
        """
        x = x.to(self.device) # set as batch, features, time in for loop below
        _, n_features, t_len = x.shape
        
        # model_x should be batch, time, features
        model_x = x.permute(0,2,1) ## CHANGE: dimensionality
        model_x = model_x.to(self.device)
        if retrospective:
            p_y_t = self.activation(self.base_model(model_x, x_padding)) ## change to be model_x

        ## CHANGE remove the loop of time here, since we are feeding in one timestep at a time instead of one batch of patients at a time
        if not retrospective:
            ## no longer need the :t+1 because I only have up to the given timestep in my model
            p_y_t = self.base_model(model_x, x_padding) ## change to be model_x; remove activation bc that's in my model
        score = torch.zeros((p_y_t.shape[0], n_features))
        for i in tqdm(range(n_features)):
            # self.data distribution shape x, feats, time; use all values for feature i to create empirical distribution
            long_feature_dist = (self.data_distribution[:, i, ind]).reshape(-1)
            feature_dist = long_feature_dist[long_feature_dist!=0].to(self.device)
            if len(feature_dist) == 0:
                print(i) # make this importance 0
                score[:, i] = 0
            else:
                x_hat = x.clone()
                kl_all=[]
                for _ in range(10):
                    ## CHANGE below so we are permuting the last time point
                    rand_idx = torch.randint(0, len(feature_dist), (len(x),), device=self.device)
                    x_hat[:, i, ind] = feature_dist[rand_idx]
                    y_hat_t = self.base_model(x_hat.permute(0,2,1), x_padding) ## CHANGE: permute
                    # kl = torch.nn.KLDivLoss(reduction='none')(torch.log(y_hat_t), p_y_t)
                    kl = torch.abs(y_hat_t - p_y_t)
                    # kl_all.append(torch.sum(kl, -1).cpu().detach().numpy())
                    kl_all.append(kl.detach())
                E_kl = torch.stack(kl_all).mean(dim=0)
                # score[:, i, t] = 2./(1+np.exp(-1*E_kl)) - 1.
                score[:, i] = E_kl.cpu()
        return score

