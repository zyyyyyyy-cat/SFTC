import json
import math
import re
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'WHU/SFTC_1000_DDP'
rows = [json.loads(s) for s in (RUN / 'sftc_diagnostics.jsonl').read_text().splitlines()]
train = [r for r in rows if r['kind'] == 'train']
clips = [r for r in rows if r['kind'] == 'epoch_clip']
log = (RUN / 'train_log_20261013_2331.txt').read_text(encoding='utf-8')
loss_rows = []
for m in re.finditer(r'Epoch\[(\d+)\] Iteration\[(\d+)/(\d+)\] Loss: ([\d.]+), Acc: ([\d.]+), Base Lr: ([\deE.+-]+)\n[^\n]*Epoch\[\1\] ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)', log):
    e, it, steps, loss, acc, lr, ce, wrt, proca, gpd = m.groups()
    loss_rows.append(dict(epoch=int(e), iteration=int(it), steps=int(steps), total=float(loss),
                          acc=float(acc), base_lr=float(lr), ce=float(ce), wrt=float(wrt),
                          proca=float(proca), gpd=float(gpd),
                          reconstruction_error=float(loss)-float(ce)-float(wrt)-.01*float(proca)-float(gpd)))

def avg(values):
    return float(np.mean(values))

def expected_lr(e, base):
    start = .01*.032
    floor = .002*.032
    return start+e*(base-start)/5 if e<5 else floor+.5*(base-floor)*(1+math.cos(math.pi*e/120))

epoch_summary = []
for e in sorted(set(r['epoch'] for r in train)):
    selected = [r for r in train if r['epoch']==e]
    finite = [r for r in selected if math.isfinite(r['global_gradient_pre_clip'])]
    loss = [r for r in loss_rows if r['epoch']==e][-1]
    bias = np.array(selected[-1]['source_bias'])
    summary = dict(epoch=e, sampled_steps=[r['iteration'] for r in selected], loss_at_100=loss,
                   bias_range=[float(bias.min()),float(bias.max())], bias_mean_per_sensor=bias.mean(1).tolist(),
                   new_lr=next(iter(selected[-1]['new_lr'].values())),
                   expected_new_lr=expected_lr(e,.096), expected_base_lr=expected_lr(e,.032),
                   cls_rho_mean={}, patch_rho_mean={}, cls_rho_at_100={}, patch_rho_at_100={},
                   clip=next(r for r in clips if r['epoch']==e),
                   global_norm_sampled=[r['global_gradient_pre_clip'] for r in selected],
                   gradient_cnn_sampled_mean=avg([r['gradient_pre_clip']['cnn'] for r in finite]),
                   gradient_kv_sampled_mean=avg([r['gradient_pre_clip']['kv'] for r in finite]),
                   cnn_relative_update_sampled_mean=avg([r['updates']['cnn']['relative_l2'] for r in finite]),
                   kv_relative_update_sampled_mean=avg([r['updates']['kv']['relative_l2'] for r in finite]))
    for sensor in ('RGB','NIR','TIR'):
        summary['cls_rho_mean'][sensor]=avg([v for r in selected for v in r['rho_local'][sensor]['cls_per_head']])
        summary['patch_rho_mean'][sensor]=avg([v for r in selected for v in r['rho_local'][sensor]['patch_per_head']])
        summary['cls_rho_at_100'][sensor]=avg(selected[-1]['rho_local'][sensor]['cls_per_head'])
        summary['patch_rho_at_100'][sensor]=avg(selected[-1]['rho_local'][sensor]['patch_per_head'])
    epoch_summary.append(summary)

issues = []
for r in train:
    e,it=r['epoch'],r['iteration']
    if r['attention_shape'] != [192,12,129,257]: issues.append(['shape',e,it])
    if r['bias_fixed'] != (e<=5): issues.append(['freeze_flag',e,it])
    if e<=5 and not np.all(np.array(r['source_bias'])==-2): issues.append(['freeze_value',e,it])
    if max(r['new_lr'].values())-min(r['new_lr'].values())>1e-10: issues.append(['lr_group_disagreement',e,it])
    if abs(next(iter(r['new_lr'].values()))-expected_lr(e,.096))>1e-10: issues.append(['lr_mismatch',e,it])
    for sensor,v in r['rho_local'].items():
        for field,p in v.items():
            if len(p)!=12 or any(not math.isfinite(x) or not 0<=x<=1 for x in p): issues.append(['rho_range',e,it,sensor,field])
    if math.isfinite(r['global_gradient_pre_clip']):
        if abs(r['clip_coefficient']-min(1,1/(r['global_gradient_pre_clip']+1e-6)))>1e-10: issues.append(['clip_mismatch',e,it])
        for part in ('cnn','kv'):
            if not 0 < r['gradient_pre_clip'][part] < float('inf'): issues.append(['bad_gradient',e,it,part])
            if r['updates'][part]['l2']<=0: issues.append(['zero_update',e,it,part])

result = dict(samples=len(train),epoch_clip_rows=len(clips),logged_loss_rows=len(loss_rows),
              issues=issues,epoch_summary=epoch_summary,
              sampled_skips=[{'epoch':r['epoch'],'iteration':r['iteration']} for r in train if r['amp_step_skipped']],
              nonfinite_sampled=[{'epoch':r['epoch'],'iteration':r['iteration']} for r in train if not math.isfinite(r['global_gradient_pre_clip'])],
              largest_loss_reconstruction_error=max(abs(r['reconstruction_error']) for r in loss_rows),
              cls_differences=[json.loads(p.read_text()) for p in sorted(RUN.glob('sftc_cls_diagnostics_rank*.jsonl'))],
              partial_validation=[{'q':int(m[0]),'g':int(m[1]),'r1':float(m[2]),'map':float(m[3]),'minp':float(m[4])}
                                  for m in re.findall(r'Pair \(Q:(\d+), G:(\d+)\) -> Rank-1: ([\d.]+)%, mAP: ([\d.]+)%, mINP: ([\d.]+)%',log)],
              has_overall_validation='Validation Results - Epoch:' in log)
(ROOT/'tmp/sftc_run_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print('issues:',issues,'samples:',len(train),'sampled skips:',result['sampled_skips'],
      'max loss reconstruction error:',result['largest_loss_reconstruction_error'])
print('ep loss acc CE WRT ProCA GPD | newLR biasMinMax | CLS RGB/NIR/TIR at100 | PATCH at100 | clipMean | cnn/kv relative update means')
for s in epoch_summary:
    r=s['loss_at_100']
    print(s['epoch'],*[round(r[k],5) for k in ('total','acc','ce','wrt','proca','gpd')],
          '|',round(s['new_lr'],6),[round(x,6) for x in s['bias_range']],
          '|',[round(x,4) for x in s['cls_rho_at_100'].values()],
          '|',[round(x,4) for x in s['patch_rho_at_100'].values()],
          '|',round(s['clip']['mean_clip_coefficient'],4),
          '|',round(s['cnn_relative_update_sampled_mean'],7),round(s['kv_relative_update_sampled_mean'],7))
print('CLS differences',result['cls_differences'])
print('validation',result['partial_validation'],'overall complete:',result['has_overall_validation'])
print('first sampled rho', {m:avg(v['cls_per_head']) for m,v in train[0]['rho_local'].items()})
print('epoch1 iter50 rho',{m:avg(v['cls_per_head']) for m,v in train[1]['rho_local'].items()})
print('epoch10 bias perhead',train[-1]['source_bias'])
print('epoch10 last gradients',train[-1]['gradient_pre_clip'],train[-1]['global_gradient_pre_clip'])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out = ROOT/'WHU/docs/SFTC_1000_DDP_analysis'
out.mkdir(exist_ok=True)
# Preserve a strict JSON report; Infinity in the source is represented as null.
def clean(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    return value
(out/'summary.json').write_text(json.dumps(clean(result),indent=2)+'\n',encoding='utf-8')
epochs = [s['epoch'] for s in epoch_summary]
fig, axes = plt.subplots(2,3,figsize=(15,8),layout='constrained')
axes = axes.ravel()
for field,label in [('total','Total'),('ce','CE'),('wrt','WRT'),('gpd','GPD')]:
    axes[0].plot(epochs,[s['loss_at_100'][field] for s in epoch_summary],marker='o',label=label)
axes[0].set_title('Loss: running mean through iteration 100')
axes[0].set_ylabel('Loss')
axes[0].legend(fontsize=9)
for i,kind in [(1,'cls_rho_at_100'),(2,'patch_rho_at_100')]:
    for sensor,color in [('RGB','#1463ac'),('NIR','#e18b1a'),('TIR','#208d62')]:
        axes[i].plot(epochs,[100*s[kind][sensor] for s in epoch_summary],color=color,marker='o',label=sensor)
    axes[i].set_title(('CLS' if i==1 else 'Patch')+' local attention mass: batch at iteration 100')
    axes[i].set_ylabel('Mean over 12 heads (%)')
    axes[i].legend(fontsize=9)
axes[3].plot(epochs,[100*s['clip']['mean_clip_coefficient'] for s in epoch_summary],marker='o',label='Mean clip coefficient')
axes[3].plot(epochs,[100*s['clip']['clip_frequency'] for s in epoch_summary],marker='s',label='Clipping frequency')
axes[3].set_ylim(0,110)
axes[3].set_title('Gradient clipping: all steps, rank 0')
axes[3].set_ylabel('Percent')
axes[3].legend(fontsize=9)
axes[4].plot(epochs,[s['loss_at_100']['proca'] for s in epoch_summary],color='#a64997',marker='o')
axes[4].set_title('Raw ProCA: running mean through iteration 100')
axes[4].set_ylabel('Loss (before weight 0.01)')
for key,label in [('cnn_relative_update_sampled_mean','CNN'),('kv_relative_update_sampled_mean','K/V')]:
    axes[5].plot(epochs,[100*s[key] for s in epoch_summary],marker='o',label=label)
axes[5].set_title('Actual relative parameter update\nper sampled step')
axes[5].set_ylabel('L2(delta) / L2(parameters) (%)')
axes[5].legend(fontsize=9)
for ax in axes:
    ax.title.set_fontsize(10)
    ax.axvline(5.5,color='gray',linestyle='--',alpha=.7)
    ax.set_xlabel('Epoch')
    ax.set_xticks(epochs)
    ax.grid(alpha=.2)
fig.suptitle('SFTC WHU-MARS-1000: early training, epochs 1-10\nDashed line: source bias becomes learnable at epoch 6; measurements are not retrieval gains',fontsize=13)
fig.savefig(out/'trends.png',dpi=170)
plt.close(fig)
print('saved',out)
