import numpy as np
import sys
import os
import pdb

class lstm_param:
  def __init__(self,input_dim,num_hidden_units):
    cat_len = input_dim + num_hidden_units
    self.Wg = np.random.randn(num_hidden_units,cat_len)
    self.Wi = np.random.randn(num_hidden_units,cat_len)
    self.Wf = np.random.randn(num_hidden_units,cat_len)
    self.Wo = np.random.randn(num_hidden_units,cat_len)
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


class lstm_states:

  def __init__(self):
    self.xc = {}
    self.g  = {}
    self.h  = {}
    self.C  = {}
    self.ig = {} 
    self.fg = {}
    self.og = {} 
    self.C[-1] = np.zeros((num_hidden_units,1))
    self.h[-1] = np.zeros((num_hidden_units,1))
 
    self.dhnext = np.zeros((num_hidden_units,1))
    self.dCnext = np.zeros((num_hidden_units,1))

  def reinit(self):
    self.xc = {}
    self.g  = {}
    self.h  = {}
    self.C  = {}
    self.ig = {}
    self.fg = {}
    self.og = {}

    self.C[-1] = np.zeros((num_hidden_units,1))
    self.h[-1] = np.zeros((num_hidden_units,1))
    self.dhnext = np.zeros((num_hidden_units,1))
    self.dCnext = np.zeros((num_hidden_units,1))


class LSTMUnit:
  
  def __init__(self,num_hidden_units,input_dim):

    self.num_hidden_units = num_hidden_units 
    self.input_dim = input_dim 
    self.lstm_param = lstm_param(input_dim,num_hidden_units)
    self.state_vars = lstm_states()  
  
  def forward(self,x,seq_id):

    if seq_id == 0:
      self.state_vars.reinit()
      self.lstm_param.reinit()

    states = self.state_vars
    states.xc[seq_id] = np.resize(np.hstack(states.h[seq_id-1].flatten(),x.flatten()),(self.input_dim+self.num_hidden_units,1))
    states.g[seq_id] = np.tanh(np.dot(self.lstm_param.Wg,self.xc[seq_id]) + self.lstm_param.bg)
    states.ig[seq_id] = sigmoid(np.dot(self.lstm_param.Wi,self.xc[seq_id]) + self.lstm_param.bi)
    states.fg[seq_id] = sigmoid(np.dot(self.lstm_param.Wf,self.xc[seq_id]) + self.lstm_param.bf)
    states.og[seq_id] = sigmoid(np.dot(self.lstm_param.Wo,self.xc[seq_id]) + self.lstm_param.bo)
    states.C[seq_id] =  states.ig[seq_id]*g[seq_id] + states.fg[seq_id]*states.C[seq_id - 1]
    states.h[seq_id] =  states.og[seq_id]*states.C[seq_id]

    return states.h[seq_id]

  def backward(self,top_diff_h,i):

    states = self.state_vars
    dh = top_diff_h + states.dHnext
    dC = dh*states.og[i] + states.dCnext

    dog = dh*states.C[i]
    dig = dC*states.g[i]
    dfg = dC*states.C[i-1]
    dg = dC*states.ig[i]

    dog_in = dog*(states.og[i]*(1.-states.og[i]))
    dig_in = dig*(states.ig[i]*(1.-states.ig[i]))
    dfg_in = dfg*(states.fg[i]*(1.-states.fg[i]))
    dg_in = dg*(1.-states.g[i]**2)

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
    states.dHnext = dXc[:self.num_hidden_units]

    return dh


input_txt = sys.argv[1]
num_lstm_layers = 2

f = open(txtFile,'r')
data = f.read()
chars = list(set(self.data))
num_data = len(self.data)
vocab_size = len(chars)
char_to_index = {c:i for i,c in enumerate(chars)}
index_to_char = {i:c for i,c in enumerate(chars)}

lstm_units = {} 

input_dim = vocab_size
num_hidden_units = 100
learning_rate = 0.1


for i in range(num_lstm_layers):
  lstm_units[i] = lstm_unit(num_hidden_units,input_dim)
  input_dim = num_hidden_units


Wy = np.random.randn(vocab_size,num_hidden_units)
by = np.zeros((vocab_size,1))

mWy = np.zeros_like(Wy)
mby = np.zeros_like(by)


if True:

  for i in range(len(data)/seq_len):
    inputs = [char_to_index[ch] for ch in lstm.data[i*seq_len:(i+1)*seq_len]]
    outputs= [char_to_index[ch] for ch in lstm.data[i*seq_len + 1:(i+1)*seq_len + 1]]
    probs = {} 
    loss = 0
    dWy = np.zeros_like(Wy)
    dby = np.zeros_like(by)

    for j in range(seq_len):
      x = np.zeros((vocab_size,1))
      x[inputs[j]] = 1
      target = outputs[j]       

      for k in range(num_lstm_layers):
        h1 = lstm_units[k].forward(x,j)
        x = np.copy(h1)

      y = np.dot(Wy,h1) + by 

      probs[j] = 1/(1+np.exp(-y))
 
      loss += -np.log(probs[j][target])     




    for j in reversed(range(seq_len)):

      dy = np.copy(probs[j])
      dy[outputs[j]] -= 1

      dWy += np.dot(dy,h[i].T)
      dby += dy

      dh = np.dot(self.Wy.T,dy) 

      for k in reversed(range(num_lstm_layers)):
        dh = lstm_units[k].backward(dh,j)

       
    mWy += (dWy)*(dWy)
    mby += (dby)*(dby)

    Wy -= learning_rate*dWy/(np.sqrt(mWy + 1e-8))
    by -= learning_rate*dby/(np.sqrt(mby + 1e-8))

    for k in range(num_lstm_layers):
      lstm_units[k].lstm_param.adagrad_step(learning_rate)















