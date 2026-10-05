"""Plot controller weights across training episodes from existing episode CSVs."""
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_training_weights(folder):
    folder=Path(folder)
    files=sorted((p for p in folder.glob('episode_*.csv') if re.fullmatch(r'episode_\d+\.csv',p.name)),key=lambda p:int(p.stem.split('_')[1]))
    if not files:raise FileNotFoundError(f'No episode_*.csv in {folder}. Use the TRAIN output folder.')
    groups={'Q':[f'q_{i}' for i in range(6)],'R':[f'r_{i}' for i in range(2)],'Qf':[f'qf_{i}' for i in range(6)]}
    cols=sum(groups.values(),[]);rows=[]
    for p in files:
        d=pd.read_csv(p,usecols=cols)
        if d.empty or not np.isfinite(d.to_numpy()).all():raise ValueError(f'Empty/nonfinite weights: {p}')
        row={'episode':int(p.stem.split('_')[1])+1,'steps':len(d)}
        for c in cols:
            for stat in ['mean','min','max']:row[f'{c}_{stat}']=float(getattr(d[c],stat)())
        rows.append(row)
    summary=pd.DataFrame(rows);summary.to_csv(folder/'training_weights_summary.csv',index=False)
    # One figure, each weight has its own scale so small changes remain visible.
    fig,axes=plt.subplots(3,6,figsize=(18,9),squeeze=False)
    labels=['vx','vy','psi','yaw rate','X','Y']
    for gi,(group,names) in enumerate(groups.items()):
        for i,ax in enumerate(axes[gi]):
            if i>=len(names):ax.set_visible(False);continue
            c=names[i];x=summary.episode.to_numpy()
            ax.fill_between(x,summary[c+'_min'].to_numpy(),summary[c+'_max'].to_numpy(),alpha=.22,color='tab:blue',label='Within-episode min-max')
            ax.plot(x,summary[c+'_mean'],color='tab:blue',lw=1.8,marker='.' if len(x)<15 else None,label='Episode mean')
            ax.set_title(f'{group}: '+(labels[i] if group!='R' else ['Delta a','Delta delta'][i]))
            ax.set_xlabel('Training episode (1-based)');ax.set_ylabel('Weight');ax.grid(alpha=.25)
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Controller weights during training: mean and within-episode range\nVariation includes learning, changing states, and exploration',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(folder/'training_weights_by_episode.png',dpi=170);plt.close(fig)
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('folder',type=Path);args=ap.parse_args()
    s=plot_training_weights(args.folder);print(f'Plotted {len(s)} episodes -> {args.folder / "training_weights_by_episode.png"}')
