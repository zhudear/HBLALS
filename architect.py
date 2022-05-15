import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    self.denoise_optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                      lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                      weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta,vgg_model,loss_fn_alex,discriminator, network_optimizer):
    loss = self.model._loss(input, target,vgg_model,loss_fn_alex,discriminator,1)[0]
    theta = _concat(self.model.net_parameters()).data#parameters
    try:#momentum*v,用的就是Network进行w更新的momentum
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.net_parameters()).mul_(self.network_momentum)#parameters
    except:
      moment = torch.zeros_like(theta)
    '''for k,v in self.model.net_named_parameters():
      print(k)'''
      #print(torch.autograd.grad(loss,v))
    dtheta = _concat(torch.autograd.grad(loss, self.model.net_parameters())).data + self.network_weight_decay*theta#parameters
    # print(eta)
    # print(moment+dtheta)
    # m_d = moment + dtheta
    # e_m_d = eta[0]*m_d
    unrolled_model = self._construct_model_from_theta(theta.sub(eta[0]*(moment + dtheta)))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta,vgg_model, loss_fn_alex,discriminator,network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:#
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta,vgg_model,loss_fn_alex,discriminator, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid,discriminator)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid,discriminator):
    loss = self.model._loss(input_valid, target_valid,discriminator,1)[0]
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train,input_valid, target_valid, eta,vgg_model,loss_fn_alex,discriminator, network_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train,  eta,vgg_model,loss_fn_alex,discriminator, network_optimizer)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid,vgg_model,loss_fn_alex,discriminator,1)[0]

    unrolled_loss.backward()
    #print('arch_parameters')
    '''for v in unrolled_model.arch_parameters():
      print(v.grad)'''#v.shape,v,v.grad.shape, torch.Size([7, 6]) torch.Size([7, 6]) torch.Size([7, 6])
    #dalpha = [torch.zeros_like(v) for v in unrolled_model.arch_parameters()]  # 对lamda求梯度
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]#none
    #print('dalpha1',dalpha)

    '''print('dalpha', len(dalpha))#dalpha 9
    for v in unrolled_model.arch_parameters():
      print(v.shape,v.requires_grad)'''
    '''
        torch.Size([8, 9]) True
torch.Size([8, 9]) True
torch.Size([8, 9]) True
torch.Size([8, 9]) True
torch.Size([8, 9]) True
torch.Size([8, 9]) True
torch.Size([8, 9]) True
torch.Size([9, 4]) True
torch.Size([9, 4]) True
'''
    '''flag=True
    for v in unrolled_model.arch_parameters():
      if v.grad is  None:
        flag=False
    if flag:
      print('dalpha not none')
    else:
      print('dalpha none')
    flag = True
    for v in unrolled_model.net_parameters():
      if v.grad is  None:
        flag = False
    if flag:
      print('vector not none')
    else:
      print('vector none')'''
    vector = [v.grad.data for v in unrolled_model.net_parameters()]#parameters
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train,vgg_model,loss_fn_alex,discriminator)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta[0]*ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)
    #print('dalpha',dalpha)

  def _construct_model_from_theta(self, theta):
      model_new = self.model.new()
      params, offset = {}, 0
      '''for k, v in self.model.net_parameters():
        print(k)'''
      for v1,v2 in zip(model_new.net_parameters(),
                        self.model.net_parameters()):  # self.model.net_parameters(): # self.model.named_parameters():
        #print(k1,k2)
        # print(k,v)#tensor([1.], device='cuda:0', requires_grad=True)
        v_length = np.prod(v2.size())
        #print(v_length,theta.shape,v1.shape,v2.shape)#2304 tensor([0.9939, 0.9983, 0.9859, 0.7901], device='cuda:0')
        #432 torch.Size([70]) torch.Size([16, 3, 3, 3]) torch.Size([16, 3, 3, 3])
        v1.data.copy_(theta[offset: offset + v_length].view(v2.size()))
        offset += v_length
      return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, vgg_model,loss_fn_alex,discriminator, r=1e-2):
    R = r / _concat(vector).norm()
    # dαLtrain(w+,α)
    for p, v in zip(self.model.net_parameters(), vector):#parameters
      #print(p.shape,v.shape,R.shape)#p v should have the same shape
      p.data.add_(R, v)
    loss = self.model._loss(input, target,vgg_model,loss_fn_alex,discriminator,1)[0]
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.net_parameters(), vector):#parameters
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target,vgg_model,loss_fn_alex,discriminator,1)[0]
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())
    # 将模型的参数从w-恢复成w
    for p, v in zip(self.model.net_parameters(), vector):#parameters
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

