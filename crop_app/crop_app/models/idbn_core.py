"""
Improved Deep Belief Network (IDBN) – Pure NumPy Implementation
Architecture (Annexure B):
  GRBM  : n_input → 256   (Gaussian-Bernoulli, handles continuous data)
  RBM-1 : 256     → 128
  RBM-2 : 128     → 64
  Output: 64      → n_classes  (classification) / 1 (regression)

Optimizer: Ranger = RAdam + Lookahead
"""

import numpy as np

# ════════════════════════════════════════════════
# Utilities
# ════════════════════════════════════════════════
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
def relu(x):    return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(float)

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(probs, y_onehot):
    return -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))

def mse(pred, target): return np.mean((pred - target)**2)


# ════════════════════════════════════════════════
# Batch Normalisation
# ════════════════════════════════════════════════
class BatchNorm:
    def __init__(self, n, eps=1e-5, momentum=0.1):
        self.gamma = np.ones(n);  self.beta = np.zeros(n)
        self.rm = np.zeros(n);    self.rv = np.ones(n)
        self.eps = eps;           self.momentum = momentum
        self.cache = None

    def forward(self, x, training=True):
        if training:
            mu = x.mean(0); var = x.var(0)
            self.rm = (1-self.momentum)*self.rm + self.momentum*mu
            self.rv = (1-self.momentum)*self.rv + self.momentum*var
        else:
            mu, var = self.rm, self.rv
        xh = (x - mu) / np.sqrt(var + self.eps)
        self.cache = (x, xh, mu, var)
        return self.gamma * xh + self.beta

    def backward(self, dout):
        x, xh, mu, var = self.cache
        N = x.shape[0]
        dgamma = (dout * xh).sum(0)
        dbeta  = dout.sum(0)
        dxh    = dout * self.gamma
        dvar   = (-0.5 * dxh * (x-mu) * (var+self.eps)**(-1.5)).sum(0)
        dmu    = (-dxh/np.sqrt(var+self.eps)).sum(0) + dvar*(-2*(x-mu)).mean(0)
        dx     = dxh/np.sqrt(var+self.eps) + dvar*2*(x-mu)/N + dmu/N
        return dx, dgamma, dbeta


# ════════════════════════════════════════════════
# Ranger Optimizer  (RAdam + Lookahead)
# ════════════════════════════════════════════════
class Ranger:
    def __init__(self, lr=1e-3, betas=(0.9,0.999), eps=1e-8, wd=0.001, k=6, alpha=0.5):
        self.lr=lr; self.b1,self.b2=betas; self.eps=eps; self.wd=wd
        self.k=k; self.alpha=alpha
        self.rho_inf = 2/(1-self.b2) - 1
        self._m={}; self._v={}; self._t={}
        self._slow={}; self._step=0

    def update(self, p, g, key):
        if self.wd: g = g + self.wd*p
        if key not in self._m:
            self._m[key]=np.zeros_like(p); self._v[key]=np.zeros_like(p); self._t[key]=0
        self._t[key] += 1; t=self._t[key]
        self._m[key] = self.b1*self._m[key] + (1-self.b1)*g
        self._v[key] = self.b2*self._v[key] + (1-self.b2)*g**2
        bc1 = 1-self.b1**t; bc2 = np.sqrt(1-self.b2**t)
        rho = self.rho_inf - 2*t*(self.b2**t)/(1-self.b2**t)
        if rho > 4:
            rect = np.sqrt((rho-4)*(rho-2)*self.rho_inf/((self.rho_inf-4)*(self.rho_inf-2)*rho))
            p = p - self.lr*rect*bc2/bc1 * self._m[key]/(np.sqrt(self._v[key])+self.eps)
        else:
            p = p - self.lr/bc1 * self._m[key]
        # Lookahead
        if key not in self._slow: self._slow[key]=p.copy()
        self._step += 1
        if self._step % (self.k * len(self._t)) == 0:
            self._slow[key] += self.alpha*(p - self._slow[key])
            p = self._slow[key].copy()
        return p


# ════════════════════════════════════════════════
# GRBM  (Gaussian-Bernoulli)
# ════════════════════════════════════════════════
class GRBM:
    def __init__(self, nv, nh):
        self.W=np.random.randn(nv,nh)*0.01; self.bv=np.zeros(nv); self.bh=np.zeros(nh)
    def h(self, v): return sigmoid(v@self.W + self.bh)
    def v(self, h): return h@self.W.T + self.bv
    def step(self, v, lr):
        h0=self.h(v); h0s=(np.random.rand(*h0.shape)<h0).astype(float)
        v1=self.v(h0s)+np.random.randn(*v.shape)*0.1; h1=self.h(v1)
        N=v.shape[0]
        self.W  += lr*(v.T@h0 - v1.T@h1)/N
        self.bh += lr*(h0-h1).mean(0); self.bv += lr*(v-v1).mean(0)
        return np.mean((v-v1)**2)
    def transform(self, v): return self.h(v)


# ════════════════════════════════════════════════
# RBM  (Bernoulli-Bernoulli)
# ════════════════════════════════════════════════
class RBM:
    def __init__(self, nv, nh):
        self.W=np.random.randn(nv,nh)*0.01; self.bv=np.zeros(nv); self.bh=np.zeros(nh)
    def h(self, v): return sigmoid(v@self.W + self.bh)
    def v(self, h): return sigmoid(h@self.W.T + self.bv)
    def step(self, v, lr):
        h0=self.h(v); h0s=(np.random.rand(*h0.shape)<h0).astype(float)
        v1=self.v(h0s); h1=self.h(v1)
        N=v.shape[0]
        self.W  += lr*(v.T@h0 - v1.T@h1)/N
        self.bh += lr*(h0-h1).mean(0); self.bv += lr*(v-v1).mean(0)
        return np.mean((v-v1)**2)
    def transform(self, v): return self.h(v)


# ════════════════════════════════════════════════
# IDBN
# ════════════════════════════════════════════════
class IDBN:
    def __init__(self, n_in=42, n_cls=12, hidden=(256,128,64), dropout=0.4, lr=0.001, wd=0.001):
        self.n_cls=n_cls; self.drop=dropout
        # RBM stack
        self.grbm=GRBM(n_in,hidden[0]); self.rbm1=RBM(hidden[0],hidden[1]); self.rbm2=RBM(hidden[1],hidden[2])
        # MLP weights (warm-started after pretrain)
        self.W1=np.random.randn(n_in,hidden[0])*0.01;     self.b1=np.zeros(hidden[0])
        self.W2=np.random.randn(hidden[0],hidden[1])*0.01; self.b2=np.zeros(hidden[1])
        self.W3=np.random.randn(hidden[1],hidden[2])*0.01; self.b3=np.zeros(hidden[2])
        self.Wc=np.random.randn(hidden[2],n_cls)*0.01;    self.bc=np.zeros(n_cls)
        self.Wr=np.random.randn(hidden[2],1)*0.01;        self.br=np.zeros(1)
        # Batch norm
        self.bn1=BatchNorm(hidden[0]); self.bn2=BatchNorm(hidden[1]); self.bn3=BatchNorm(hidden[2])
        # Optimizer
        self.opt=Ranger(lr=lr, wd=wd)
        self.y_reg_max = 1.0

    # ── Pre-train ──────────────────────────────────
    def pretrain(self, X, epochs=50, bs=32, lr=0.001, verbose=True):
        print("\n── Pre-training (CD-1) ───────────────────────────")
        data=X.copy(); N=X.shape[0]
        for rbm, name in [(self.grbm,'GRBM'),(self.rbm1,'RBM-1'),(self.rbm2,'RBM-2')]:
            if verbose: print(f"  {name}  {data.shape[1]}→{rbm.W.shape[1]}")
            for ep in range(epochs):
                idx=np.random.permutation(N); loss=0
                for i in range(0,N,bs):
                    loss += rbm.step(data[idx[i:i+bs]], lr)
                if verbose and (ep+1)%10==0:
                    print(f"    ep {ep+1:3d}/{epochs}  recon={loss/(N//bs):.4f}")
            data=rbm.transform(data)
        # Warm-start MLP
        self.W1[:]=self.grbm.W; self.b1[:]=self.grbm.bh
        self.W2[:]=self.rbm1.W; self.b2[:]=self.rbm1.bh
        self.W3[:]=self.rbm2.W; self.b3[:]=self.rbm2.bh
        print("  Done.\n")

    # ── Forward ────────────────────────────────────
    def _fwd(self, X, train=False):
        def drp(a):
            if train and self.drop>0:
                m=(np.random.rand(*a.shape)>self.drop).astype(float)
                return a*m/(1-self.drop), m
            return a, None
        z1=X@self.W1+self.b1; a1=self.bn1.forward(z1,train); h1=relu(a1); h1,m1=drp(h1)
        z2=h1@self.W2+self.b2; a2=self.bn2.forward(z2,train); h2=relu(a2); h2,m2=drp(h2)
        z3=h2@self.W3+self.b3; a3=self.bn3.forward(z3,train); h3=relu(a3); h3,m3=drp(h3)
        logits=h3@self.Wc+self.bc; probs=softmax(logits)
        reg=(h3@self.Wr+self.br).squeeze(-1) if h3.ndim>1 else (h3@self.Wr+self.br).squeeze()
        return probs, reg, (X,z1,a1,h1,m1,z2,a2,h2,m2,z3,a3,h3,m3)

    # ── Backward ───────────────────────────────────
    def _bwd(self, probs, reg, cache, yoh, yr_n, rw=0.1):
        X,z1,a1,h1,m1,z2,a2,h2,m2,z3,a3,h3,m3 = cache
        N=X.shape[0]
        dl=((probs-yoh)/N)
        dWc=h3.T@dl; dbc=dl.sum(0); dh3c=dl@self.Wc.T
        dr=(reg-yr_n)*2*rw/N
        dWr=h3.T@dr.reshape(-1,1); dbr=np.array([dr.sum()])
        dh3r=dr.reshape(-1,1)@self.Wr.T
        dh3=dh3c+dh3r
        if m3 is not None: dh3*=m3/(1-self.drop)
        da3=dh3*relu_grad(a3); da3b,dg3,db3b=self.bn3.backward(da3)
        dW3=h2.T@da3b; db3=da3b.sum(0); dh2=da3b@self.W3.T
        if m2 is not None: dh2*=m2/(1-self.drop)
        da2=dh2*relu_grad(a2); da2b,dg2,db2b=self.bn2.backward(da2)
        dW2=h1.T@da2b; db2=da2b.sum(0); dh1=da2b@self.W2.T
        if m1 is not None: dh1*=m1/(1-self.drop)
        da1=dh1*relu_grad(a1); da1b,dg1,db1b=self.bn1.backward(da1)
        dW1=X.T@da1b; db1=da1b.sum(0)
        return dict(W1=dW1,b1=db1,W2=dW2,b2=db2,W3=dW3,b3=db3,
                    Wc=dWc,bc=dbc,Wr=dWr,br=dbr,
                    g1=dg1,b1n=db1b,g2=dg2,b2n=db2b,g3=dg3,b3n=db3b)

    def _apply(self, grads):
        for k in ['W1','b1','W2','b2','W3','b3','Wc','bc','Wr','br']:
            setattr(self, k, self.opt.update(getattr(self,k), grads[k], k))
        for bn,g,b,sfx in [(self.bn1,'g1','b1n','_1'),(self.bn2,'g2','b2n','_2'),(self.bn3,'g3','b3n','_3')]:
            bn.gamma=self.opt.update(bn.gamma,grads[g],'gm'+sfx)
            bn.beta =self.opt.update(bn.beta, grads[b],'bt'+sfx)

    # ── Fit ────────────────────────────────────────
    def fit(self, Xtr, yc, yr, Xv, ycv, yrv, epochs=100, bs=32, pt_epochs=50, verbose=True):
        self.y_reg_max = yr.max() + 1e-8
        self.pretrain(Xtr, epochs=pt_epochs, bs=bs, verbose=verbose)
        N=Xtr.shape[0]; nc=self.n_cls
        yoh=np.eye(nc)[yc]; yr_n=yr/self.y_reg_max
        best_acc=0; best=None; patience=15; no_imp=0
        hist={'train_loss':[],'val_loss':[],'val_acc':[]}
        print("── Fine-tuning (Ranger) ──────────────────────────")
        for ep in range(epochs):
            idx=np.random.permutation(N); tl=0
            for i in range(0,N,bs):
                b=idx[i:i+bs]
                p,r,cache=self._fwd(Xtr[b],train=True)
                loss=cross_entropy(p,yoh[b])+0.1*mse(r,yr_n[b])
                grads=self._bwd(p,r,cache,yoh[b],yr_n[b])
                self._apply(grads); tl+=loss
            vp,vr,_=self._fwd(Xv,train=False)
            vl=cross_entropy(vp,np.eye(nc)[ycv])
            acc=(vp.argmax(1)==ycv).mean()*100
            hist['train_loss'].append(tl/(N//bs))
            hist['val_loss'].append(float(vl))
            hist['val_acc'].append(acc)
            if acc>best_acc: best_acc=acc; best=self._state(); no_imp=0
            else: no_imp+=1
            if verbose and (ep+1)%10==0:
                print(f"  ep {ep+1:3d}  train={tl/(N//bs):.4f}  val={vl:.4f}  acc={acc:.1f}%")
            if no_imp>=patience:
                if verbose: print(f"  Early stop ep {ep+1}")
                break
        self._load(best); print(f"\n  Best acc: {best_acc:.2f}%")
        return hist

    def predict(self, X):
        p,_,_=self._fwd(X,train=False); return p.argmax(1)
    def predict_proba(self, X):
        p,_,_=self._fwd(X,train=False); return p
    def predict_yield(self, X):
        _,r,_=self._fwd(X,train=False); return np.maximum(0,r*self.y_reg_max)
    def _state(self):
        return {k:getattr(self,k).copy() for k in ['W1','b1','W2','b2','W3','b3','Wc','bc','Wr','br']}
    def _load(self, s):
        for k,v in s.items(): setattr(self,k,v.copy())
