import numpy as np
from torch import nn


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;self.net=nn.Sequential(nn.Linear(c.get('input_shape',[8])[-1],32),nn.SiLU(),nn.Linear(32,c.get('programs',6)))
    def forward(self,b):return {'probability':self.net(b['x'].float()).softmax(-1)}
    def objective(self,b,p):return -(p['probability']*b['compatibility'].float()).sum(-1).clamp_min(1e-8).log().mean()


def demo(seed,n,c):
    rng=np.random.default_rng(seed);x=rng.normal(size=(n,8)).astype('float32');y=np.argmax(x[:,:c.get('programs',6)],1);compat=np.eye(c.get('programs',6))[y]
    return {'x':x,'y':y,'compatibility':compat.astype('float32'),'group':np.arange(n)}


def entropy(probability):
    p=np.asarray(probability);return float(-(p[p>0]*np.log(p[p>0])).sum())


def search(problem,variant='full'):
    answers=np.asarray(problem['answers']);weight=np.asarray(problem.get('prior',np.ones(len(answers))),float);weight/=weight.sum();probes=list(problem.get('initial_probes',[0]));queried=[];executions=len(answers)*len(probes);label_cost=0.;history=[]
    target=problem['target_program'];noise=problem.get('label_noise',0.)
    def partition():
        classes={}
        for p in np.flatnonzero(weight>0):classes.setdefault(tuple(answers[p,probes]),[]).append(int(p))
        return list(classes.values())
    while label_cost<problem['label_budget']:
        classes=partition();mass=np.array([weight[ids].sum() for ids in classes]);total=entropy(weight);exposed=entropy(mass);hidden=total-exposed
        history.append({'exposed_entropy':exposed,'hidden_entropy':hidden,'total_entropy':total,'retained_programs':int(np.sum(weight>0)),'execution_cost':executions,'label_cost':label_cost})
        if total<1e-10:break
        if variant not in {'baseline','fixed_probes'} and exposed<.5*total:
            candidate=None
            for z in range(answers.shape[1]):
                if z in probes or executions+len(answers)>problem['execution_budget']:continue
                executions+=len(answers);gain=0.
                for ids,m in zip(classes,mass):
                    labels=np.unique(answers[ids,z]);prob=[weight[np.array(ids)[answers[ids,z]==label]].sum()/max(m,1e-12) for label in labels];gain+=m*entropy(prob)
                if candidate is None or gain>candidate[0]:candidate=(gain,z)
            if candidate:probes.append(candidate[1]);classes=partition()
        available=[z for z in probes if z not in queried and label_cost+problem['query_cost'][z]<=problem['label_budget']]
        if not available:break
        utilities=[]
        for z in available:
            distribution=[weight[answers[:,z]==label].sum() for label in np.unique(answers[:,z])];utilities.append(entropy(distribution)/problem['query_cost'][z])
        z=available[int(np.argmax(utilities))];label=answers[target,z];queried.append(z);label_cost+=problem['query_cost'][z]
        likelihood=np.where(answers[:,z]==label,1-noise,noise/max(len(np.unique(answers[:,z]))-1,1));weight*=likelihood
        if not weight.sum():raise ValueError('Observed labels are inconsistent with every retained program')
        weight/=weight.sum()
        if variant=='representative_only':
            for ids in partition():weight[ids[0]]=weight[ids].sum();weight[ids[1:]]=0
    return {'posterior':weight.tolist(),'probes':probes,'queried':queried,'execution_cost':executions,'label_cost':label_cost,'history':history,'target_mass':float(weight[target])}


def execute(args):
    import json
    from pathlib import Path
    result=search(json.loads(Path(args.input).read_text()),args.variant);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'search.json').write_text(json.dumps(result,indent=2))
