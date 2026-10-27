import json
import math
import re
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'WHU/SFTC_1000_DDP'
OUT = ROOT/'WHU/docs/SFTC_1000_DDP_complete_analysis'
OUT.mkdir(exist_ok=True)

def parse_log(name):
    text = (RUN/name).read_text(encoding='utf-8')
    config_text = re.split(r'\n\d{4}-\d{2}-\d{2} ',text.split('Running with config:\n',1)[1],1)[0]
    config = yaml.safe_load(config_text)
    train = {}
    pattern = r'Epoch\[(\d+)\] Iteration\[(\d+)/(\d+)\] Loss: ([\d.]+), Acc: ([\d.]+), Base Lr: ([\deE.+-]+)\n[^\n]*Epoch\[\1\] ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)'
    for m in re.finditer(pattern,text):
        e,it,steps,total,acc,lr,ce,wrt,proca,gpd=m.groups()
        if int(it)==100:
            train[int(e)]={'total':float(total),'acc':float(acc),'base_lr_printed':float(lr),
                           'ce':float(ce),'wrt':float(wrt),'proca':float(proca),'gpd':float(gpd),'steps':int(steps)}
    evals=[]
    for m in re.finditer(r'Validation Results - Epoch: (\d+)\n[^\n]*mAP: ([\d.]+)%\n[^\n]*Rank-1\s*: ([\d.]+)%\n[^\n]*Rank-5\s*: ([\d.]+)%\n[^\n]*Rank-10\s*: ([\d.]+)%',text):
        e,ap,r1,r5,r10=m.groups()
        evals.append({'epoch':int(e),'map':float(ap),'r1':float(r1),'r5':float(r5),'r10':float(r10)})
    starts=[m.end() for m in re.finditer('Starting modality-pair specific evaluation.',text)]
    pairs=[]
    ag=[]
    for start,ev in zip(starts,evals):
        end=text.index('Validation Results - Epoch:',start)
        block=text[start:end]
        pair={}
        for m in re.finditer(r'Pair \(Q:(\d+), G:(\d+)\) -> Rank-1: ([\d.]+)%, mAP: ([\d.]+)%, mINP: ([\d.]+)%',block):
            q,g,r1,ap,inp=m.groups()
            pair[q+'->'+g]={'r1':float(r1),'map':float(ap),'minp':float(inp)}
        pairs.append({'epoch':ev['epoch'],'pairs':pair})
        directions={}
        for m in re.finditer(r'(A->G|G->A) -> mAP: ([\d.]+)%, Rank-1: ([\d.]+)%, Rank-10: ([\d.]+)%',block):
            key,ap,r1,r10=m.groups()
            directions[key]={'map':float(ap),'r1':float(r1),'r10':float(r10)}
        ag.append({'epoch':ev['epoch'],'directions':directions})
    timing={int(e):{'seconds_batch':float(t),'seconds_epoch':float(tt)}
            for e,t,tt in re.findall(r'Epoch (\d+) done. Time per batch: ([\d.]+)\[s\] Total: ([\d.]+)\[s\]',text)}
    return dict(config=config,train=train,evals=evals,pairs=pairs,ag=ag,timing=timing)

uad=parse_log('UADtrain_log_20260911_2145.txt')
sftc=parse_log('train_log_20261013_2331.txt')
for data in (uad,sftc):
    assert set(data['train'])==set(range(1,121))
    assert set(data['timing'])==set(range(1,121))
    assert [r['epoch'] for r in data['evals']]==list(range(10,121,10))
    assert [r['epoch'] for r in data['pairs']]==list(range(10,121,10))
    assert all(set(r['pairs'])=={f'{q}->{g}' for q in range(1,4) for g in range(1,4)} for r in data['pairs'])
    assert all(set(r['directions'])=={'A->G','G->A'} for r in data['ag'])
diff=[]
for section in set(uad['config'])|set(sftc['config']):
    a,b=uad['config'].get(section),sftc['config'].get(section)
    if isinstance(a,dict) and isinstance(b,dict):
        for key in set(a)|set(b):
            if a.get(key)!=b.get(key): diff.append([section+'.'+key,a.get(key),b.get(key)])
    elif a!=b: diff.append([section,a,b])

rows=[json.loads(x) for x in (RUN/'sftc_diagnostics.jsonl').read_text().splitlines()]
tr=[r for r in rows if r['kind']=='train']
clips=[r for r in rows if r['kind']=='epoch_clip']
summaries=[]
for epoch in range(1,121):
    records=[r for r in tr if r['epoch']==epoch]
    finite=[r for r in records if math.isfinite(r['global_gradient_pre_clip'])]
    bias=np.array(records[-1]['source_bias'])
    entry={'epoch':epoch,'bias_range':[float(bias.min()),float(bias.max())],
           'bias_mean_per_sensor':bias.mean(1).tolist(),
           'new_lr':next(iter(records[-1]['new_lr'].values())),
           'global_gradient_mean':float(np.mean([r['global_gradient_pre_clip'] for r in finite])),
           'gradients':{k:float(np.mean([r['gradient_pre_clip'][k] for r in finite])) for k in ('cnn','kv')},
           'relative_updates':{k:float(np.mean([r['updates'][k]['relative_l2'] for r in finite])) for k in ('cnn','kv')},
           'absolute_updates':{k:float(np.mean([r['updates'][k]['l2'] for r in finite])) for k in ('cnn','kv')},
           'clip':next(r for r in clips if r['epoch']==epoch),'rho':{}}
    for sensor in ('RGB','NIR','TIR'):
        entry['rho'][sensor]={kind:float(np.mean([v for r in records for v in r['rho_local'][sensor][kind]]))
                              for kind in ('cls_per_head','patch_per_head')}
    summaries.append(entry)
cls=[json.loads(line) for p in sorted(RUN.glob('sftc_cls_diagnostics_rank*.jsonl')) for line in p.read_text().splitlines()]
assert len(tr)==360 and len(clips)==120 and len(cls)==36
assert all(sorted(r['iteration'] for r in tr if r['epoch']==e)==[1,50,100] for e in range(1,121))
issues=[]
for r in tr:
    if r['attention_shape'] != [192,12,129,257]: issues.append(['shape',r['epoch'],r['iteration']])
    if r['bias_fixed'] != (r['epoch']<=5): issues.append(['freeze',r['epoch'],r['iteration']])
    if r['epoch']<=5 and not np.all(np.array(r['source_bias'])==-2): issues.append(['fixed_bias',r['epoch'],r['iteration']])
    for sensor,p in r['rho_local'].items():
        if any(not math.isfinite(x) or not 0<=x<=1 for v in p.values() for x in v): issues.append(['rho',r['epoch'],r['iteration'],sensor])
    if not r['amp_step_skipped']:
        if any(not math.isfinite(v) or v<=0 for v in r['gradient_pre_clip'].values()): issues.append(['grad',r['epoch'],r['iteration']])
        if any(v['l2']<=0 or not math.isfinite(v['l2']) for v in r['updates'].values()): issues.append(['update',r['epoch'],r['iteration']])

comparison=[]
for a,b in zip(uad['evals'],sftc['evals']):
    assert a['epoch']==b['epoch']
    comparison.append({'epoch':a['epoch'],'uad':a,'sftc':b,'delta':{k:b[k]-a[k] for k in ('map','r1','r5','r10')}})
result={'uad':uad,'sftc':sftc,'config_differences':diff,'validation_comparison':comparison,
        'diagnostics_by_epoch':summaries,'cls_differences':cls,
        'sampled_nonfinite':[{'epoch':r['epoch'],'iteration':r['iteration']} for r in tr if not math.isfinite(r['global_gradient_pre_clip'])],
        'sampled_amp_skips':[{'epoch':r['epoch'],'iteration':r['iteration']} for r in tr if r['amp_step_skipped']],
        'issues':issues,'diagnostic_sample_count':len(tr),'training_epoch_count':[len(uad['train']),len(sftc['train'])],
        'sampler_steps_equal_all_epochs':all(uad['train'][e]['steps']==sftc['train'][e]['steps'] for e in range(1,121)),
        'logged_base_lr_equal_all_epochs':all(uad['train'][e]['base_lr_printed']==sftc['train'][e]['base_lr_printed'] for e in range(1,121))}
loss_error=max(abs(r['total']-r['ce']-r['wrt']-.01*r['proca']-r['gpd'])
               for data in (uad,sftc) for r in data['train'].values())
assert loss_error<=.00201
result['analysis_checks']={'counts_and_epochs_valid':True,'weighted_loss_max_rounding_error':loss_error,
                          'pair_map_lower_count':sum(b['pairs'][p]['map']<a['pairs'][p]['map']
                              for a,b in zip(uad['pairs'],sftc['pairs']) for p in a['pairs']),
                          'total_steps_per_run':sum(r['steps'] for r in sftc['train'].values())}
(OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print('config differences',diff)
print('issues',issues,'sampled skips',result['sampled_amp_skips'],'all steps equal',result['sampler_steps_equal_all_epochs'])
print('EPOCH | UAD mAP/R1 | SFTC mAP/R1 | differences')
for r in comparison: print(r['epoch'],r['uad']['map'],r['uad']['r1'],'|',r['sftc']['map'],r['sftc']['r1'],'|',round(r['delta']['map'],2),round(r['delta']['r1'],2))
print('EPOCH | UAD and SFTC [loss,acc,CE,WRT,ProCA,GPD] | bias range | cls and patch RGB/NIR/TIR | clip | newLR')
for e in [1,2,5,6,10,20,30,40,50,60,80,100,120]:
    d=summaries[e-1]
    print(e,[uad['train'][e][k] for k in ['total','acc','ce','wrt','proca','gpd']],
          [sftc['train'][e][k] for k in ['total','acc','ce','wrt','proca','gpd']],
          '|',[round(x,4) for x in d['bias_range']],
          '|',[round(d['rho'][m]['cls_per_head'],4) for m in ['RGB','NIR','TIR']],
          [round(d['rho'][m]['patch_per_head'],4) for m in ['RGB','NIR','TIR']],
          '|',round(d['clip']['mean_clip_coefficient'],3),'|',round(d['new_lr'],6))
print('FINAL PAIRS UAD SFTC delta')
for key,a in uad['pairs'][-1]['pairs'].items():
    b=sftc['pairs'][-1]['pairs'][key]
    print(key,a,b,{k:round(b[k]-a[k],2) for k in a})
print('FINAL AG',uad['ag'][-1],sftc['ag'][-1])
print('FINAL CLS',[r for r in cls if r['epoch']==120])
print('BEST UAD',max(uad['evals'],key=lambda x:x['map']),'BEST SFTC',max(sftc['evals'],key=lambda x:x['map']))
print('TIMING',[{'model':name,'train_seconds':sum(t['seconds_epoch'] for t in data['timing'].values()),
                 'late_batch_median':float(np.median([t['seconds_batch'] for e,t in data['timing'].items() if e>=60]))}
                for name,data in [('UAD',uad),('SFTC',sftc)]])

# Reconstruct only the aggregate pre-step parameter norm from recorded deltas.
# A norm shrinking like decay does not identify the direction of learned updates.
regularization=[]
w_new=v_new=0.
w_new=1.
w_base=1.
v_base=0.
for e in range(1,121):
    r=[r for r in tr if r['epoch']==e][-1]
    lr=next(iter(r['new_lr'].values()))
    base_lr=.00032+e*(.032-.00032)/5 if e<5 else .000064+.5*(.032-.000064)*(1+math.cos(math.pi*e/120))
    for it in range(sftc['train'][e]['steps']):
        if e==1 and it==0: continue
        v_new=.9*v_new+.0001*w_new
        w_new-=lr*v_new
        v_base=.9*v_base+.0001*w_base
        w_base-=base_lr*v_base
    regularization.append({'epoch':e,'pure_decay_multiplier_new':w_new,'pure_decay_multiplier_base':w_base,
                           'aggregate_pre_step_norm':{p:r['updates'][p]['l2']/r['updates'][p]['relative_l2'] for p in ('cnn','kv')}})
result['regularization_reference']={'assumptions':'zero data gradient, scalar SGD momentum0.9/decay1e-4, same LR schedule and actual epoch step counts, skips only the observed initial step; does not simulate actual learned parameters',
                                    'by_epoch':regularization}
(OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors={'uad':'#1463ac','sftc':'#d76b22'}
epochs=list(range(1,121))
fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
axes=axes.ravel()
for model,data in [('uad',uad),('sftc',sftc)]:
    e=[r['epoch'] for r in data['evals']]
    axes[0].plot(e,[r['map'] for r in data['evals']],marker='o',label=model.upper(),color=colors[model])
    axes[1].plot(e,[r['r1'] for r in data['evals']],marker='o',label=model.upper(),color=colors[model])
    axes[2].plot(epochs,[data['train'][e]['total'] for e in epochs],label=model.upper(),color=colors[model])
    axes[3].plot(epochs,[100*data['train'][e]['acc'] for e in epochs],label=model.upper(),color=colors[model])
    axes[4].plot(epochs,[data['train'][e]['proca'] for e in epochs],label=model.upper(),color=colors[model])
axes[5].plot(epochs,[s['rho']['TIR']['cls_per_head']*100 for s in summaries],color='#208d62',label='TIR CLS local mass')
axes[5].plot(epochs,[s['rho']['RGB']['cls_per_head']*100 for s in summaries],color='#1463ac',label='RGB CLS local mass')
axes[5].plot(epochs,[s['rho']['NIR']['cls_per_head']*100 for s in summaries],color='#e18b1a',label='NIR CLS local mass')
titles=['AS-ReID validation mAP','AS-ReID validation Rank-1','Training total loss (mean through iteration 100)',
        'Training accuracy (mean through iteration 100)','Raw ProCA (mean through iteration 100)',
        'Local attention: sparse-batch/head means']
for i,ax in enumerate(axes):
    ax.set_title(titles[i],fontsize=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Percent' if i in (0,1,3,5) else 'Loss')
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
axes[2].set_yscale('log')
fig.suptitle('UAD vs SFTC: same logged training protocol, one seed per model',fontsize=14)
fig.savefig(OUT/'comparison.png',dpi=170)
plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
axes=axes.ravel()
for sensor,color in [('RGB','#1463ac'),('NIR','#e18b1a'),('TIR','#208d62')]:
    for i,key in [(0,'cls_per_head'),(1,'patch_per_head')]:
        axes[i].plot(epochs,[s['rho'][sensor][key]*100 for s in summaries],color=color,label=sensor)
    selected=[r for r in cls if r['folder']=={'RGB':'RGB','NIR':'IR','TIR':'Thermal'}[sensor]]
    axes[2].plot([r['epoch'] for r in selected],[r['cosine_distance_mean'] for r in selected],color=color,marker='o',label=sensor)
for key,label in [('cnn','CNN'),('kv','K/V')]:
    axes[3].plot(epochs,[s['gradients'][key] for s in summaries],label=label)
    axes[4].plot(epochs,[s['absolute_updates'][key]/s['new_lr'] for s in summaries],label=label)
axes[5].plot(epochs,[r['aggregate_pre_step_norm']['kv']/regularization[0]['aggregate_pre_step_norm']['kv'] for r in regularization],label='Observed aggregate K/V norm')
axes[5].plot(epochs,[r['pure_decay_multiplier_new']/regularization[0]['pure_decay_multiplier_new'] for r in regularization],linestyle='--',label='Zero-data-gradient decay reference')
titles=['CLS local mass: sparse-batch/head means','Patch local mass: sparse-batch/head means',
        'CLS on/off cosine distance: 8 queries per sensor','Pre-clip gradient norm: finite sampled steps',
        'Actual update L2 / current LR: sampled steps','K/V norm shrinkage vs pure-decay reference']
for i,ax in enumerate(axes):
    ax.set_title(titles[i],fontsize=10)
    ax.set_xlabel('Epoch')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
axes[0].set_ylabel('Percent');axes[1].set_ylabel('Percent')
fig.suptitle('SFTC mechanism evidence: active local source; norm shrinkage is not a measure of identity utility',fontsize=13)
fig.savefig(OUT/'mechanisms.png',dpi=170)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
for ax,key,title in zip(axes,['map','r1'],['Final modality-pair mAP change','Final modality-pair Rank-1 change']):
    arr=np.array([[sftc['pairs'][-1]['pairs'][str(q)+'->'+str(g)][key]-uad['pairs'][-1]['pairs'][str(q)+'->'+str(g)][key]
                   for g in range(1,4)] for q in range(1,4)])
    im=ax.imshow(arr,cmap='RdBu',vmin=-4.5,vmax=4.5)
    for q in range(3):
        for g in range(3): ax.text(g,q,f'{arr[q,g]:+.2f} pp',ha='center',va='center',color='white' if abs(arr[q,g])>2.5 else 'black')
    ax.set_xticks(range(3),['RGB','NIR','TIR']);ax.set_yticks(range(3),['RGB','NIR','TIR'])
    ax.set_xlabel('Gallery');ax.set_ylabel('Query');ax.set_title(title,fontsize=11)
fig.colorbar(im,ax=axes,label='SFTC minus UAD (percentage points)',shrink=.8)
fig.suptitle('Epoch 120: all nine modality-pair mAP values decrease',fontsize=13)
fig.savefig(OUT/'domain_deltas.png',dpi=180)
plt.close(fig)
print('Saved figures and strict JSON:',OUT)
