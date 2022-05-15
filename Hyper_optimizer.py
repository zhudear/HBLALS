import torch
import numpy as np
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class HyperOptimizer(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.hyper_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, vgg_model,loss_fn_alex,discriminator,network_optimizer):
    loss = self.model._loss(input, target,vgg_model,loss_fn_alex,discriminator,1)[0]
    theta = _concat(self.model.net_parameters()).data#parameters
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.net_parameters()).mul_(self.network_momentum)#parameters
    except:
      moment = torch.zeros_like(theta)
    #allow_unused设为True
    #print('###########')
    '''m=[]
    for v in torch.autograd.grad(loss, self.model.net_parameters(),allow_unused=True):
      #print(v)
      if v!=None:
        m.append(v)'''
    ##here needs

    #dtheta = _concat(torch.autograd.grad(loss, self.model.net_parameters())).data + self.network_weight_decay*theta#parameters

    grad_=torch.autograd.grad(loss, self.model.net_parameters())
    #print('grad_grad_',type(grad_[0]),grad_[0].shape,len(grad_))#636
    dtheta = _concat(grad_).data + self.network_weight_decay*theta#parameters
    #nn.utils.clip_grad_norm(unrolled_model.hyper_parameters(), 5)
    #dtheta = _concat(m).data + self.network_weight_decay*theta#parameters

    #m=theta.sub(eta, moment+dtheta)
    #print('_compute_unrolled_model',m.shape,moment.shape,theta.shape)
    unrolled_model = self._construct_model_from_theta(theta.sub(eta[0]*(moment + dtheta)))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, vgg_model,loss_fn_alex,discriminator,network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta,vgg_model,loss_fn_alex,discriminator, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid,vgg_model,loss_fn_alex)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid,vgg_model,loss_fn_alex):
    loss  = self.model._loss(input_valid, target_valid,vgg_model,loss_fn_alex,2)[0]
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, vgg_model,loss_fn_alex,discriminator,network_optimizer):
    # 计算公式六：dαLval(w',α) ，其中w' = w − ξ*dwLtrain(w, α)
    # w'
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, vgg_model,loss_fn_alex,discriminator,network_optimizer)
    # Lval
    unrolled_loss = unrolled_model._loss(input_valid, target_valid,vgg_model,loss_fn_alex,discriminator,2)[0]
    #print('@@@@@@@@@@@@@@@@',input_valid.shape, target_valid.shape)
    # 对做了一次更新后的w的unrolled_model求验证集的损失，Lval，以用来对α进行更新
    unrolled_loss.backward()
    #2021.09.18 add   nn.utils.clip_grad_norm(unrolled_model.hyper_parameters(), 5)
    #nn.utils.clip_grad_norm(unrolled_model.hyper_parameters(), 5)
    # dαLval(w',α)
    #print('hyper_parameters')
    '''for v in unrolled_model.hyper_parameters():
      print(v.shape,v,v.grad)'''
    '''Parameter containing:
tensor([1.], device='cuda:0', requires_grad=True) None
'''
    #dalpha = [v.grad for v in unrolled_model.hyper_parameters()]#对lamda求梯度
    '''flag = True
    for v in unrolled_model.hyper_parameters():
      if v.grad is None:
        flag = False
    if flag:
      dalpha = [v.grad for v in unrolled_model.hyper_parameters()]
      #print('dalpha not none')
    else:
      dalpha = [torch.zeros_like(v) for v in unrolled_model.hyper_parameters()]  # 对lamda求梯度'''

    dalpha = []
    for v in unrolled_model.hyper_parameters():
      if v.grad is None:
        dalpha.append(torch.zeros_like(v))
      else:
        dalpha.append(v.grad)

      #print('dalpha none')
    # flag = True
    # for v in unrolled_model.net_parameters():
    #   if v.grad is None:
    #     flag = False
    # if flag:
    #   print('vector not none')
    # else:
    #   print('vector none')

    #dalpha = [v.grad for v in unrolled_model.hyper_parameters()]
    #dalpha = [torch.zeros_like(v) for v in unrolled_model.hyper_parameters()]  # 对lamda求梯度
    #print('dalpha', dalpha)#[None, None, None, None]
    # dw'Lval(w',α) vector就是dw'Lval(w',α)

    #here needs
    vector = [v.grad.data for v in unrolled_model.net_parameters()]#parameters
    #print('vector', vector[0])
    # unrolled_model.parameters()得到w‘
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train,vgg_model,loss_fn_alex,discriminator)
    # 公式六减公式八
    for g, ig in zip(dalpha, implicit_grads):
      #print('-',g.data)
      g.data.sub_(eta[0]*ig.data)
      #print('#', g.data)
    # 对α进行更新
    for v, g in zip(self.model.hyper_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new_hyper()
    #model_dict = self.model.state_dict()
    #,v,v.requires_grad)

    params, offset = {}, 0
    #for v in model_new.net_parameters():
    for v1, v2 in zip(model_new.net_parameters(),self.model.net_parameters()):#self.model.net_parameters(): # self.model.named_parameters():
      #print(k,v)#tensor([1.], device='cuda:0', requires_grad=True)
      v_length = np.prod(v2.size())
      #print(v_length,theta.shape,v.shape)#2304 tensor([0.9939, 0.9983, 0.9859, 0.7901], device='cuda:0')
      v1.data.copy_(theta[offset: offset+v_length].view(v2.size()))
      offset += v_length

    assert offset == len(theta)
    #model_dict.update(params)
    #model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target,vgg_model,loss_fn_alex,discriminator, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.net_parameters(), vector):#parameters
      p.data.add_(R, v)
    loss = self.model._loss(input, target,vgg_model,loss_fn_alex,discriminator,1)[0]
    #print('hyters',self.model.hyper_parameters())
    grads_p = torch.autograd.grad(loss, self.model.hyper_parameters())#,allow_unused=True)classerv5


    for p, v in zip(self.model.net_parameters(), vector):#parameters
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target,vgg_model,loss_fn_alex,discriminator,1)[0]
    grads_n = torch.autograd.grad(loss, self.model.hyper_parameters())
    #print('lamda',grads_p,grads_n)

    for p, v in zip(self.model.net_parameters(), vector):#parameters
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

