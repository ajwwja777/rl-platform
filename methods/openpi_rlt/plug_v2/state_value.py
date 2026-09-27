"""State-value model for AWBC advantages: V(s) = P(success | Stage-1 features, state).

Trained as a classifier (class-balanced BCE, standardized inputs, dropout, weight decay) with early
stopping on episodes held out from the *training* split, so the release validation split stays untouched.
There is no action input: V cannot rate the actor's own actions above the data, which is how the
action-conditioned critic failed. Advantages come from value change: A = V(s') - V(s), and at a
terminal decision A = outcome - V(s).
"""
import hashlib
import numpy as np
import torch

def inner_split(groups,frac=.15,salt='value-v1'):
    """Deterministic episode-level holdout inside the training split: True = held out for early stopping."""
    def held(g):return int(hashlib.sha256((salt+':'+str(g)).encode()).hexdigest(),16)%10000<frac*10000
    groups=np.asarray(groups);keys=sorted(set(groups.tolist()));out={g:held(g) for g in keys}
    if keys and not any(out.values()):out[keys[-1]]=True      # always hold out at least one episode
    if keys and all(out.values()):out[keys[0]]=False
    return np.array([out[g] for g in groups.tolist()])

def class_weights(y,positive_rate):
    """Balanced BCE weights: each class contributes half the loss, as the replay sampler balances outcomes."""
    p=float(np.clip(positive_rate,1e-3,1-1e-3));return np.where(np.asarray(y)>.5,.5/p,.5/(1-p)).astype(np.float32)

def fit_value(x,y,groups,*,seed=0,epochs=40,patience=6,hidden=256,lr=1e-3,weight_decay=1e-2,dropout=.2,batch=128,threads=4):
    x=np.asarray(x,np.float32);y=np.asarray(y,np.float32);hold=inner_split(groups)
    fit,val=np.flatnonzero(~hold),np.flatnonzero(hold)
    if not len(fit) or not len(val):raise ValueError('value model needs training and held-out episodes')
    torch.set_num_threads(threads);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    mu=x[fit].mean(0);sd=x[fit].std(0)+1e-4;X=torch.tensor((x-mu)/sd);Y=torch.tensor(y);W=torch.tensor(class_weights(y,y[fit].mean()))
    net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],hidden),torch.nn.ReLU(),torch.nn.Dropout(dropout),torch.nn.Linear(hidden,1))
    opt=torch.optim.AdamW(net.parameters(),lr,weight_decay=weight_decay)
    def loss(idx):
        j=torch.tensor(idx);return (torch.nn.functional.binary_cross_entropy_with_logits(net(X[j]).squeeze(-1),Y[j],reduction='none')*W[j]).sum()/W[j].sum()
    best=(float('inf'),-1,None);history=[]
    for epoch in range(epochs):
        net.train();order=fit.copy();rng.shuffle(order)
        for b in range(0,len(order),batch):
            l=loss(order[b:b+batch]);opt.zero_grad();l.backward();opt.step()
        net.eval()
        with torch.no_grad():v=float(loss(val))
        history.append(v)
        if v<best[0]:best=(v,epoch,{k:t.clone() for k,t in net.state_dict().items()})
        elif epoch-best[1]>=patience:break
    net.load_state_dict(best[2]);net.eval()
    model={'net':net,'mu':mu,'sd':sd}
    report={'best_epoch':best[1],'epochs_run':len(history),'inner_val_loss':best[0],'fit_rows':int(len(fit)),'inner_val_rows':int(len(val)),
            'inner_val_episodes':int(len(set(np.asarray(groups)[hold].tolist()))),'positive_rate':float(y[fit].mean())}
    return model,report

def predict(model,x,batch=4096):
    x=(np.asarray(x,np.float32)-model['mu'])/model['sd'];out=[]
    with torch.no_grad():
        for b in range(0,len(x),batch):out.append(torch.sigmoid(model['net'](torch.tensor(x[b:b+batch]))).squeeze(-1).numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0,np.float32)

def advantages(v,v_next,done,outcome):
    """A = V(s') - V(s); at a terminal decision the next value is the observed outcome."""
    v=np.asarray(v,np.float32);return np.where(np.asarray(done,bool),np.asarray(outcome,np.float32),np.asarray(v_next,np.float32))-v
