import torch as t
from torch.optim import Optimizer
from torch.autograd import grad

def isn(a):
	return t.mean(t.isnan(a).float())

def trans(a):
	b = t.sign(a)
	return -0.5*b**2 + 0.5*b + 1

class CurveBall(Optimizer):
	"""
		CurveBall optimizer
		extending from the Optimizer class of pytorch
	"""
	def __init__(self, params, auto_lambda=True, lambd=10.0,
			lambda_factor=0.999, lambda_low=0.5, lambda_high=1.5, lambda_interval=5,eps=1e-4):
		"""
			the standard init function this just sets the default dictionary
			params - model params 
			auto_lambda - lambda tuning
			lambda_* - all are for lambda tuning
			lambd - the actual lambda value
			beta,rho - are just initialised they have their own closed form solution
			eps - added for numerical stability
		"""
		defaults = dict(betaz=1e-3, rhoz=0.9, betas=1e-3, rhos=0.9, auto_lambda=auto_lambda,
			lambd=lambd, lambda_factor=lambda_factor, lambda_low=lambda_low,
		lambda_high=lambda_high, lambda_interval=lambda_interval,eps=eps)
		super().__init__(params, defaults)

	def step(self, model_fn, loss_fn):
		"""
			Performs a single optimization step
			Most important part everything goes on here
			supports only one param_group which is okay
			since, no tunable params except lambda
			param_groups - list of param_group
			state - a dictionary of state tensors - not sure ?
		"""
		# only support one parameter group
		if len(self.param_groups) != 1:
			raise ValueError('Since the hyper-parameters are set automatically, only one parameter group (with the same hyper-parameters) is supported.')
		group = self.param_groups[0]
		parameters = group['params']

		state = self.state
		for p in parameters:
			if p not in state:
				state[p] = {'z': t.zeros_like(p), 's': t.zeros_like(p)}
		
		zs = [state[p]['z'] for p in parameters]
		ss = [state[p]['s'] for p in parameters]
		
		global_state = state[parameters[0]]
		global_state.setdefault('count', 0)

		lambd = global_state.get('lambd', group['lambd'])
		eps = global_state.get('eps', group['eps'])
		
		"""
			Forward prop - step1
		"""
		predictions = model_fn()
		predictions_d = predictions.detach().requires_grad_(True)
		loss = loss_fn(predictions_d)
		
		"""
			Calculating the delta_z
		"""
		(Jz,) = fmad(predictions, parameters, zs)  # equivalent but slower
		# compute loss gradient Jl, retaining the graph to allow 2nd-order gradients
		(Jl,) = grad(loss, predictions_d, create_graph=True)
		Jl_d = Jl.detach()  # detached version, without requiring gradients
		# compute loss Hessian (projected by Jz) using 2nd-order gradients
		(Hl_Jz,) = grad(Jl, predictions_d, grad_outputs=Jz, retain_graph=True)
		# compute J * (Hl_Jz + Jl) using RMAD (back-propagation)
		delta_zs = grad(predictions, parameters, Hl_Jz + Jl_d, retain_graph=True)
		J_z = grad(predictions, parameters, Jl_d, retain_graph=True)
		# add lambda * z term to the result, obtaining the final steps delta_zs
		for (z, dz) in zip(zs, delta_zs):
			dz.data.add_(lambd, z)

		"""
			Precomputations for s before actually evaluating
			the functional value and jacobian
		"""
		(Js,) = fmad(predictions, parameters, ss)  # equivalent but slower	
		# compute loss Hessian (projected by Js) using 2nd-order gradients
		(Hl_Js,) = grad(Jl, predictions_d, grad_outputs=Js, retain_graph=True)
		# compute J * (Hl_Js) using RMAD (back-propagation).
		# note this is still missing the lambda * s + loss jacobian or ostrowski's term .***
		delta_ss = grad(predictions, parameters, Hl_Js, retain_graph=True)
		# this is for the computation of hyper parameters 
		# we use current gradient as an approximation for the next gradient
		delta_s1 = grad(predictions, parameters, Hl_Js+Jl_d, retain_graph=True)

		for (s, ds) in zip(ss, delta_s1):
			pre_ds = ds.clone()
			ds.data.add_(lambd, s)

		
		"""
			Computing automatic hyper parameters for step1 or z
			here, we also compute some parts required for the next step
		"""
		beta_z, rho_z = group['betaz'], group['rhoz']

		if rho_z < 0 or beta_z < 0 or group['auto_lambda']:
			# compute J^T * delta_zs
			(Jdeltaz,) = fmad(predictions, parameters, delta_zs)  # equivalent but slower
			# project result by loss hessian (using 2nd-order gradients)
			(Hl_Jdeltaz,) = grad(Jl, predictions_d, grad_outputs=Jdeltaz, retain_graph=True)
			# solve 2x2 linear system: [rho, -beta]^T = [a11, a12; a12, a22]^-1 [b1, b2]^T.
			# accumulate components of dot-product from all parameters, by first aggregating them into a vector.
			z_vec = t.cat([z.flatten() for z in zs])
			dz_vec = t.cat([dz.flatten() for dz in delta_zs])
			
			a11 = lambd * (dz_vec * dz_vec).sum() + (Jdeltaz * Hl_Jdeltaz).sum()
			a12 = lambd * (dz_vec * z_vec).sum() + (Jz * Hl_Jdeltaz).sum()
			a22 = lambd * (z_vec * z_vec).sum() + (Jz * Hl_Jz).sum()
			b1 = (Jl_d * Jdeltaz).sum()
			b2 = (Jl_d * Jz).sum()
			A = t.tensor([[a11, a12], [a12, a22]])
			b = t.tensor([[b1], [b2]])
			auto_params = t.matmul(A.pinverse(), b)

			beta_z = auto_params[0].item()
			rho_z = -auto_params[1].item()
			# compute the terms required for other steps
			(Jdeltas,) = fmad(predictions, parameters, delta_s1)
			(Hl_Jdeltas,) = grad(Jl, predictions_d, grad_outputs=Jdeltas)

		"""
			parameter updatation for z
		"""
		for (p, z, dz) in zip(parameters, zs, delta_zs):
			z.data.mul_(rho_z).add_(-beta_z, dz)
			p.data.add_(z)

		"""
			computations for step2
			forward and then calculation for ostrowski or any-other method
		"""
		predictions2 = model_fn()
		predictions2_d = predictions2.detach().requires_grad_(True)
		loss2 = loss_fn(predictions2_d)
		
		"""
			evaluation for delta_ss
		"""
		# now we need to calculate Jl2
		(Jl2,) = grad(loss2, predictions2_d, create_graph=True)
		Jl2_d = Jl2.detach()

		# now calculate Jl2Jphi2
		J_s = grad(predictions2, parameters, Jl2_d)

		# now calculate the term for ostrwoski and update them
		for (dz, ds) in zip(J_z, J_s):
			den = dz-2*ds
			pre_ds = ds.clone()
			ds.mul_(dz/(den - trans(den)*eps))

		# addition with delta_ss
		for (ds,j) in zip(delta_ss, J_s):
			ds.data.add_(j.detach())

		# add lambda * s term to the result, obtaining the final steps delta_ss
		for (s, ds) in zip(ss, delta_ss):
			ds.data.add_(lambd, s)
		
		"""
			finding hyper parameters for s
		"""
		beta_s, rho_s = group['betas'], group['rhos']

		if beta_s < 0 or rho_s < 0 or group['auto_lambda']:  # required by auto-lambda
			# use previously computed Jdeltas and Hl_Jdeltas with approximation
			# solve 2x2 linear system: [rho, -beta]^T = [a11, a12; a12, a22]^-1 [b1, b2]^T.
			# accumulate components of dot-product from all parameters, by first aggregating them into a vector.
			
			s_vec = t.cat([s.flatten() for s in ss])
			ds_vec = t.cat([ds.flatten() for ds in delta_ss])

			a11 = lambd * (ds_vec * ds_vec).sum() + (Jdeltas * Hl_Jdeltas).sum()
			a12 = lambd * (ds_vec * s_vec).sum() + (Js * Hl_Jdeltas).sum()
			a22 = lambd * (s_vec * s_vec).sum() + (Js * Hl_Js).sum()

			# for calculating b
			for (ds, dj, s) in zip (J_s, delta_ss, s):
				b1+=t.sum(ds*dj).item()
				b2+=t.sum(ds*s).item()
			# b1 = (Jl2_d * Jdeltass).sum()
			# b2 = (Jl2_d * Js2).sum()

			A = t.tensor([[a11, a12], [a12, a22]])
			b = t.tensor([[b1], [b2]])
			auto_params = t.matmul(A.pinverse(), b)

			beta_s = auto_params[0].item()
			rho_s = -auto_params[1].item()

		"""
			Parameter update
		"""
		for (p, s, ds) in zip(parameters, ss, delta_ss):
			s.data.mul_(rho_s).add_(-beta_s, ds)  # update state
			p.data.add_(s)

		"""
			lambda-tuning
		"""
		if group['auto_lambda']:
			# only adapt once every few batches
			if global_state['count'] % group['lambda_interval'] == 0:
				with t.no_grad():
					# evaluate the loss with the updated parameters
					new_loss = loss_fn(model_fn())	
					# objective function change predicted by quadratic fit
					quadratic_change = -0.5 * (auto_params * b).sum()
					# ratio between predicted and actual change
					ratio = (new_loss - loss) / quadratic_change
					# increase or decrease lambda based on ratio
					factor = group['lambda_factor'] ** group['lambda_interval']
					if ratio < group['lambda_low']: lambd /= factor
					if ratio > group['lambda_high']: lambd *= factor
					global_state['lambd'] = lambd

		global_state['count'] += 1
		return (loss, predictions)


def fmad(ys, xs, dxs):
	"""
		Forward-mode automatic differentiation.
		grad(outputs, inputs, grad_outputs)
	"""
	v = t.zeros_like(ys, requires_grad=True)
	g = grad(ys, xs, grad_outputs=v, create_graph=True)
	return grad(g, v, grad_outputs=dxs)
