import numpy as np
from algo import ValueFunctionWithApproximation
import torch
import torch.nn as nn

class ValueFunctionWithNN(ValueFunctionWithApproximation, torch.nn.Module):
	def __init__(self, state_dims):
        #state_dims: the number of dimensions of state space
        
        # TODO: implement this method
		super().__init__()
		self.net = nn.Sequential(
		            nn.Linear(state_dims, 32),
		            nn.ReLU(),
		            nn.Linear(32, 32),
		            nn.ReLU(),
		            nn.Linear(32, 1))
		#Use AdamOptimizer 
		self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999))
		self.loss_function = torch.nn.MSELoss()    	

	def __call__(self,s):
		# TODO: implement this method
		s = torch.from_numpy(s).float()
		self.net.eval()
		value=self.net(s)
		return value.detach().numpy()[0]

	def update(self,alpha,G,s_tau):
		# TODO: implement this method
		s_tau = torch.from_numpy(s_tau).float()
		pred = self.net(s_tau)
        #Set it back to train mode with model.train()
		self.net.train()
		G = torch.tensor(float(G))
		loss = self.loss_function(pred,G)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()