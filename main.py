import numpy as np
import sys
import os
import pdb
from random import uniform

def sigmoid(in1):
  out1 = 1/(1+np.exp(-in1))
  return out1


class lstm_param:
  def __init__(self,input_dim,num_hidden_units):
    cat_len = input_dim + num_hidden_units
    self.Wg = np.random.randn(num_hidden_units,cat_len)*0.01
    self.Wi = np.random.randn(num_hidden_units,cat_len)*0.01
    self.Wf = np.random.randn(num_hidden_units,cat_len)*0.01
    self.Wo = np.random.randn(num_hidden_units,cat_len)*0.01
    self.bg = np.zeros((num_hidden_units,1))
    self.bi = np.zeros((num_hidden_units,1))
    self.bf = np.zeros((num_hidden_units,1))
    self.bo = np.zeros((num_hidden_units,1))

    self.dWg = np.zeros_like(self.Wg)
    self.dWi = np.zeros_like(self.Wi)
    self.dWf = np.zeros_like(self.Wf)
    self.dWo = np.zeros_like(self.Wo)
    self.dbg = np.zeros_like(self.bg)
    self.dbi = np.zeros_like(self.bi)
    self.dbf = np.zeros_like(self.bf)
    self.dbo = np.zeros_like(self.bo)

    self.mWg = np.zeros_like(self.Wg)
    self.mWi = np.zeros_like(self.Wi)
    self.mWf = np.zeros_like(self.Wf)
    self.mWo = np.zeros_like(self.Wo)
    self.mbg = np.zeros_like(self.bg)
    self.mbi = np.zeros_like(self.bi)
    self.mbf = np.zeros_like(self.bf)
    self.mbo = np.zeros_like(self.bo)


  def reinit(self):

    self.dWg = np.zeros_like(self.Wg)
    self.dWi = np.zeros_like(self.Wi)
    self.dWf = np.zeros_like(self.Wf)
    self.dWo = np.zeros_like(self.Wo)
    self.dbg = np.zeros_like(self.bg)
    self.dbi = np.zeros_like(self.bi)
    self.dbf = np.zeros_like(self.bf)
    self.dbo = np.zeros_like(self.bo)



  def adagrad_step(self,learning_rate):


    for param in [self.dWg, self.dWi, self.dWf, self.dWo, self.dbg, self.dbi, self.dbf, self.dbo]:
      np.clip(param,-5,5,out=param)  


    self.mWg += self.dWg*self.dWg
    self.mWi += self.dWi*self.dWi
    self.mWf += self.dWf*self.dWf
    self.mWo += self.dWo*self.dWo
    self.mbg += self.dbg*self.dbg
    self.mbi += self.dbi*self.dbi
    self.mbf += self.dbf*self.dbf
    self.mbo += self.dbo*self.dbo

    
    self.Wg -= learning_rate*self.dWg/(np.sqrt(self.mWg + 1e-8))
    self.Wi -= learning_rate*self.dWi/(np.sqrt(self.mWi + 1e-8))
    self.Wf -= learning_rate*self.dWf/(np.sqrt(self.mWf + 1e-8))
    self.Wo -= learning_rate*self.dWo/(np.sqrt(self.mWo + 1e-8))
    self.bg -= learning_rate*self.dbg/(np.sqrt(self.mbg + 1e-8))
    self.bi -= learning_rate*self.dbi/(np.sqrt(self.mbi + 1e-8))
    self.bf -= learning_rate*self.dbf/(np.sqrt(self.mbf + 1e-8))
    self.bo -= learning_rate*self.dbo/(np.sqrt(self.mbo + 1e-8))
    
    """
    self.Wg -= learning_rate*self.dWg
    self.Wi -= learning_rate*self.dWi
    self.Wf -= learning_rate*self.dWf
    self.Wo -= learning_rate*self.dWo
    self.bg -= learning_rate*self.dbg
    self.bi -= learning_rate*self.dbi
    self.bf -= learning_rate*self.dbf
    self.bo -= learning_rate*self.dbo
    """





class lstm_states:

  def __init__(self,num_hidden_units):
    self.xc = {}
    self.g  = {}
    self.h  = {}
    self.C  = {}
    self.ig = {} 
    self.fg = {}
    self.og = {} 
    self.cprev = np.zeros((num_hidden_units,1))
    self.hprev = np.zeros((num_hidden_units,1))
 
    self.dhnext = np.zeros((num_hidden_units,1))
    self.dCnext = np.zeros((num_hidden_units,1))

  def reinit(self,num_hidden_units):
    self.xc = {}
    self.g  = {}
    self.h  = {}
    self.C  = {}
    self.ig = {}
    self.fg = {}
    self.og = {}
    
    self.dhnext = np.zeros((num_hidden_units,1))
    self.dCnext = np.zeros((num_hidden_units,1))


class LSTMUnit:
  
  def __init__(self,num_hidden_units,input_dim):

    self.num_hidden_units = num_hidden_units 
    self.input_dim = input_dim 
    self.lstm_param = lstm_param(input_dim,num_hidden_units)
    self.state_vars = lstm_states(num_hidden_units)  
 
  def forward(self,x,seq_id,hprev,cprev):

    if seq_id == 0:
      self.state_vars.reinit(self.num_hidden_units)
      self.lstm_param.reinit()
      self.state_vars.C[-1] = np.copy(cprev)
      self.state_vars.h[-1] = np.copy(hprev)

    states = self.state_vars
    states.xc[seq_id] = np.resize(np.hstack((states.h[seq_id-1].flatten(),x.flatten())),(self.input_dim+self.num_hidden_units,1))
    states.g[seq_id] = np.tanh(np.dot(self.lstm_param.Wg,states.xc[seq_id]) + self.lstm_param.bg)
    states.ig[seq_id] = sigmoid(np.dot(self.lstm_param.Wi,states.xc[seq_id]) + self.lstm_param.bi)
    states.fg[seq_id] = sigmoid(np.dot(self.lstm_param.Wf,states.xc[seq_id]) + self.lstm_param.bf)
    states.og[seq_id] = sigmoid(np.dot(self.lstm_param.Wo,states.xc[seq_id]) + self.lstm_param.bo)
    states.C[seq_id] =  states.ig[seq_id]*states.g[seq_id] + states.fg[seq_id]*states.C[seq_id - 1]
    states.h[seq_id] =  states.og[seq_id]*np.tanh(states.C[seq_id])


    return states.h[seq_id],states.C[seq_id]

  def backward(self,top_diff_h,i):

    states = self.state_vars
    dh = top_diff_h + states.dhnext
    C_1 = (1-(np.tanh(states.C[i]))**2)

    dC = dh*states.og[i]*C_1 + states.dCnext

    dog = dh*np.tanh(states.C[i])
    dig = dC*states.g[i]
    dfg = dC*states.C[i-1]
    dg = dC*states.ig[i]

    dog_in = dog*(states.og[i]*(1.-states.og[i]))
    dig_in = dig*(states.ig[i]*(1.-states.ig[i]))
    dfg_in = dfg*(states.fg[i]*(1.-states.fg[i]))
    dg_in = dg*(1. - states.g[i]**2)

    self.lstm_param.dWi += np.dot(dig_in,states.xc[i].T)
    self.lstm_param.dWf += np.dot(dfg_in,states.xc[i].T)
    self.lstm_param.dWo += np.dot(dog_in,states.xc[i].T)
    self.lstm_param.dWg += np.dot(dg_in,states.xc[i].T)

    self.lstm_param.dbo += dog_in
    self.lstm_param.dbi += dig_in
    self.lstm_param.dbf += dfg_in
    self.lstm_param.dbg += dg_in

    dXc = np.dot(self.lstm_param.Wg.T,dg_in)
    dXc += np.dot(self.lstm_param.Wf.T,dfg_in)
    dXc += np.dot(self.lstm_param.Wi.T,dig_in)
    dXc += np.dot(self.lstm_param.Wo.T,dog_in)

    states.dCnext = dC*states.fg[i]
    states.dhnext = dXc[:self.num_hidden_units]
    dx = dXc[self.num_hidden_units:]

    return dx


def get_loss(inputs,outputs,Wy,by,lstm_units,vocab_size,seq_len,hprev,cprev):
  
  loss = 0
  for i in range(seq_len):
    idx = inputs[i]
    x = np.zeros((vocab_size,1))
    x[idx] = 1
    for j in range(len(lstm_units)):
      h,c = lstm_units[j].forward(x,i,hprev[j],cprev[j])
      x = np.copy(h)

    y = np.dot(Wy,h) + by
    probs = np.exp(y)/sum(np.exp(y))
    loss += -np.log(probs[outputs[i]])


  return loss


def forward_backward(inputs,outputs,Wy,by,lstm_units,seq_len,vocab_size,hprev1,cprev1):

  probs = {}
  loss = 0
  dWy = np.zeros_like(Wy)
  dby = np.zeros_like(by)
  hprev = {}
  cprev = {}   
  num_lstm_layers = len(lstm_units)

  for j in range(seq_len):
    x = np.zeros((vocab_size,1))
    x[inputs[j]] = 1
    target = outputs[j]

    for k in range(num_lstm_layers):
      h1,c1 = lstm_units[k].forward(x,j,hprev1[k],cprev1[k])
      x = np.copy(h1)
      hprev[k] = np.copy(h1)
      cprev[k] = np.copy(c1)
      

    y = np.dot(Wy,h1) + by

    probs[j] = np.exp(y)/sum(np.exp(y))

    loss += -np.log(probs[j][target])

  #pdb.set_trace()

  for j in reversed(range(seq_len)):

    dy = np.copy(probs[j])
    dy[outputs[j]] -= 1

    dWy += np.dot(dy,lstm_units[num_lstm_layers-1].state_vars.h[j].T)
    dby += dy

    dh = np.dot(Wy.T,dy)

    for k in reversed(range(num_lstm_layers)):
      dh = lstm_units[k].backward(dh,j)


  return dWy,dby,loss,hprev,cprev



def sample(id1,Wy,by,lstm_units,vocab_size,index_char,seq_len,hprev,cprev):
  idx = id1
  seq = []
  seq.append(index_char[idx])
 
  for i in range(seq_len):
    x = np.zeros((vocab_size,1))
    x[idx] = 1
    for j in range(len(lstm_units)):
      h,c = lstm_units[j].forward(x,i,hprev[j],cprev[j])
      x = np.copy(h)

    y = np.dot(Wy,h) + by
    probs = np.exp(y)/sum(np.exp(y))
    idx = np.random.choice(range(vocab_size),p=probs.ravel())
    seq.append(index_char[idx])

  return seq  


def gradCheck(inputs,target,Wy,by,lstm_units,seq_len,vocab_size,hprev,cprev):

  num_checks, delta = 10, 1e-5
  dWy,dby,loss,_,_ = forward_backward(inputs,target,Wy,by,lstm_units,seq_len,vocab_size,hprev,cprev)
  idx = 0
  for param,dparam,name in zip([Wy,lstm_units[idx].lstm_param.Wg, lstm_units[idx].lstm_param.Wi, lstm_units[idx].lstm_param.Wf, lstm_units[idx].lstm_param.Wo,by,lstm_units[idx].lstm_param.bg,lstm_units[idx].lstm_param.bi,lstm_units[idx].lstm_param.bf,lstm_units[idx].lstm_param.bo], [dWy,lstm_units[idx].lstm_param.dWg, lstm_units[idx].lstm_param.dWi,lstm_units[idx].lstm_param.dWf, lstm_units[idx].lstm_param.dWo,dby,lstm_units[idx].lstm_param.dbg,lstm_units[idx].lstm_param.dbi,lstm_units[idx].lstm_param.dbf,lstm_units[idx].lstm_param.dbo], ['Wy', 'Wg-0', 'Wi-0', 'Wf-0', 'Wo-0','by','bg-0','bi-0','bf-0','bo-0']):

    s0 = dparam.shape
    s1 = param.shape
    assert s0 == s1, 'Error dims dont match: %s and %s.' % (`s0`, `s1`)
    print name
    if True:
      for i in xrange(num_checks):
        ri = int(uniform(0,param.size))
        # evaluate cost at [x + delta] and [x - delta]
        old_val = param.flat[ri]
        param.flat[ri] = old_val + delta

        cg0 = get_loss(inputs,target,Wy,by,lstm_units,vocab_size,seq_len,hprev,cprev)

        param.flat[ri] = old_val - delta

        cg1 = get_loss(inputs,target,Wy,by,lstm_units,vocab_size,seq_len,hprev,cprev)

        param.flat[ri] = old_val # reset old value for this parameter
        # fetch both numerical and analytic gradient
        grad_analytic = dparam.flat[ri]
        grad_numerical = (cg0 - cg1) / ( 2 * delta )
        rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
        print '%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error)
        # rel_error should be on order of 1e-7 or less



def gradCheck_full(inputs,target,Wy,by,lstm_units,seq_len,vocab_size,hprev,cprev):

  num_checks, delta = 10, 1e-5
  dWy,dby,loss,_,_ = forward_backward(inputs,outputs,Wy,by,lstm_units,seq_len,vocab_size,hprev,cprev)

  for param,dparam,name in zip([Wy,lstm_units[0].lstm_param.Wg, lstm_units[0].lstm_param.Wi, lstm_units[0].lstm_param.Wf, lstm_units[0].lstm_param.Wo, lstm_units[1].lstm_param.Wg,lstm_units[1].lstm_param.Wi,lstm_units[1].lstm_param.Wf,lstm_units[1].lstm_param.Wo,by,lstm_units[0].lstm_param.bg,lstm_units[0].lstm_param.bi,lstm_units[0].lstm_param.bf,lstm_units[0].lstm_param.bo,lstm_units[1].lstm_param.bg,lstm_units[1].lstm_param.bi,lstm_units[1].lstm_param.bf,lstm_units[1].lstm_param.bo], [dWy,lstm_units[0].lstm_param.dWg, lstm_units[0].lstm_param.dWi,lstm_units[0].lstm_param.dWf, lstm_units[0].lstm_param.dWo, lstm_units[1].lstm_param.dWg,lstm_units[1].lstm_param.dWi,lstm_units[1].lstm_param.dWf,lstm_units[1].lstm_param.dWo,dby,lstm_units[0].lstm_param.dbg,lstm_units[0].lstm_param.dbi,lstm_units[0].lstm_param.dbf,lstm_units[0].lstm_param.dbo,lstm_units[1].lstm_param.dbg,lstm_units[1].lstm_param.dbi,lstm_units[1].lstm_param.dbf,lstm_units[1].lstm_param.dbo], ['Wy', 'Wg-0', 'Wi-0', 'Wf-0', 'Wo-0','Wg-1','Wi-1','Wf-1','Wo-1','by','bg-0','bi-0','bf-0','bo-0','bg-1','bi-1','bf-1','bo-1']):
    
    s0 = dparam.shape
    s1 = param.shape
    assert s0 == s1, 'Error dims dont match: %s and %s.' % (`s0`, `s1`)
    print name
    if True:
      for i in xrange(num_checks):
        ri = int(uniform(0,param.size))
        # evaluate cost at [x + delta] and [x - delta]
        old_val = param.flat[ri]
        param.flat[ri] = old_val + delta

        cg0 = get_loss(inputs,target,Wy,by,lstm_units,vocab_size,seq_len,hprev,cprev)

        param.flat[ri] = old_val - delta

        cg1 = get_loss(inputs,target,Wy,by,lstm_units,vocab_size,seq_len,hprev,cprev)

        param.flat[ri] = old_val # reset old value for this parameter
        # fetch both numerical and analytic gradient
        grad_analytic = dparam.flat[ri]
        grad_numerical = (cg0 - cg1) / ( 2 * delta )
        rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
        print '%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error)
        # rel_error should be on order of 1e-7 or less


def train(txtFile,num_lstm_layers,seq_len,num_hidden_units,learning_rate):


  f = open(txtFile,'r')
  data = f.read()
  chars = list(set(data))
  num_data = len(data)
  vocab_size = len(chars)
  char_to_index = {c:i for i,c in enumerate(chars)}
  index_to_char = {i:c for i,c in enumerate(chars)}

  lstm_units = {} 

  input_dim = vocab_size


  for i in range(num_lstm_layers):
    lstm_units[i] = LSTMUnit(num_hidden_units,input_dim)
    input_dim = num_hidden_units


  Wy = np.random.randn(vocab_size,num_hidden_units)*0.01
  by = np.zeros((vocab_size,1))*0.01

  mWy = np.zeros_like(Wy)
  mby = np.zeros_like(by)


  count = 0

  i=0
  inputs = [char_to_index[ch] for ch in data[i*seq_len:(i+1)*seq_len]]
  outputs= [char_to_index[ch] for ch in data[i*seq_len + 1:(i+1)*seq_len + 1]]

  hprev = {}
  cprev = {}

  for i in range(num_lstm_layers):
    hprev[i] = np.zeros((num_hidden_units,1))
    cprev[i] = np.zeros((num_hidden_units,1))


  gradCheck(inputs,outputs,Wy,by,lstm_units,seq_len,vocab_size,hprev,cprev)


  #pdb.set_trace()

  loss1 = 0

  while True:

    for i in range(num_lstm_layers):
      hprev[i] = np.zeros((num_hidden_units,1))
      cprev[i] = np.zeros((num_hidden_units,1))


    for i in range(len(data)/seq_len):
      inputs = [char_to_index[ch] for ch in data[i*seq_len:(i+1)*seq_len]]
      outputs= [char_to_index[ch] for ch in data[i*seq_len + 1:(i+1)*seq_len + 1]]

      dWy,dby,loss,hprev,cprev = forward_backward(inputs,outputs,Wy,by,lstm_units,seq_len,vocab_size,hprev,cprev)
     
      #pdb.set_trace()
 
      np.clip(dWy,-5,5,out=dWy)
      np.clip(dby,-5,5,out=dby)

 
      mWy += (dWy)*(dWy)
      mby += (dby)*(dby)

      Wy -= learning_rate*dWy/(np.sqrt(mWy + 1e-8))
      by -= learning_rate*dby/(np.sqrt(mby + 1e-8))

      #Wy -= learning_rate*dWy
      #by -= learning_rate*dby


      #pdb.set_trace()

      #print 'loss =',loss

      for k in range(num_lstm_layers):
        lstm_units[k].lstm_param.adagrad_step(learning_rate)

      if count%100 == 0:

        #gradCheck(inputs,outputs,Wy,by,lstm_units,seq_len,vocab_size,hprev,cprev)
        #pdb.set_trace()
        seq = sample(char_to_index[data[(i+1)*seq_len + 1]],Wy,by,lstm_units,vocab_size,index_to_char,100,hprev,cprev)
        txt = ''.join(ix for ix in seq)
        
        print txt + ' END ' 

      count += 1

      #if count%1000 == 0:
      #  gradCheck(inputs,outputs,Wy,by,lstm_units,seq_len,vocab_size,hprev,cprev)


txtFile1 = sys.argv[1]
train(txtFile1,2,50,128,0.1)




