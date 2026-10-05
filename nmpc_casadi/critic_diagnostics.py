"""Frozen-policy diagnostics for V=-b-e'Qf e. Quadratic bands are not full value bounds."""
import numpy as np
import pandas as pd


def calculate(logs, gamma):
    d=pd.DataFrame(logs).copy()
    terminal=bool(d.terminated.iloc[-1])
    tail=0. if terminal else float(d.next_value.iloc[-1])
    returns=np.empty(len(d))
    for k in range(len(d)-1,-1,-1):
        tail=float(d.reward.iloc[k])+gamma*tail
        returns[k]=tail
    d['discounted_return']=returns
    d['return_is_complete']=terminal
    d['value_error']=d.value-returns
    d['td_target']=d.reward+gamma*(1-d.terminated)*d.next_value
    d['td_error']=d.td_target-d.value
    tol=1e-9*(1+np.abs(returns))
    d['return_below_quadratic_min']=returns<d.quadratic_min-tol
    d['return_above_quadratic_max']=returns>d.quadratic_max+tol
    # Since b>=0 has no fixed upper bound, the full architecture has no finite lower bound.
    d['return_above_full_value_upper']=returns>d.full_value_upper+tol
    d['quadratic_value']=d.value+d.value_b
    d['region']=np.where(np.abs(d.curvature)>=.05,'curve','straight')
    progress=d.progress_before_m if 'progress_before_m' in d else d.progress_m
    d.loc[progress>=progress.iloc[-1]-.2,'region']='last_0.2m_observed'
    return d


def summarize(d):
    return dict(samples=len(d),return_is_complete=bool(d.return_is_complete.all()),
        value_mae=float(np.abs(d.value_error).mean()),value_rmse=float(np.sqrt(np.mean(d.value_error**2))),
        value_bias=float(d.value_error.mean()),td_rms=float(np.sqrt(np.mean(d.td_error**2))),
        return_outside_quadratic_range_pct=float(100*(d.return_below_quadratic_min|d.return_above_quadratic_max).mean()),
        return_below_quadratic_min_pct=float(100*d.return_below_quadratic_min.mean()),
        return_above_full_value_upper_pct=float(100*d.return_above_full_value_upper.mean()),
        baseline_b_mean=float(d.value_b.mean()),baseline_b_max=float(d.value_b.max()),
        baseline_share_of_value_pct=float(100*d.value_b.mean()/max(-d.value.mean(),1e-12)))


def export_diagnostics(logs,folder,scenario,gamma):
    import matplotlib.pyplot as plt
    d=calculate(logs,gamma)
    d.to_csv(folder/f'{scenario}_critic_diagnostics.csv',index=False)
    rows=[{'region':'all',**summarize(d)}]
    rows.extend({'region':key,**summarize(group)} for key,group in d.groupby('region'))
    pd.DataFrame(rows).to_csv(folder/f'{scenario}_critic_summary.csv',index=False)
    fig,axes=plt.subplots(4,1,figsize=(11,12),sharex=True)
    axes[0].fill_between(d.t_s,d.quadratic_min,d.quadratic_max,alpha=.2,label='Quadratic-only range (b=0)')
    axes[0].plot(d.t_s,d.value,label='Full critic V(s)')
    axes[0].plot(d.t_s,d.discounted_return,label='Observed discounted return G' if d.return_is_complete.all() else 'Return with critic tail (truncated)')
    axes[0].set_ylabel('Value / return')
    axes[1].plot(d.t_s,d.value_b,label='Nonnegative baseline b(s)')
    axes[1].plot(d.t_s,-d.quadratic_value,label='Quadratic cost e^T Qf e')
    axes[1].set_ylabel('Cost components')
    axes[2].plot(d.t_s,d.value_error,label='V - G')
    axes[2].plot(d.t_s,d.td_error,label='One-step TD error')
    axes[2].axhline(0,color='black',linewidth=.6);axes[2].set_ylabel('Error')
    axes[3].plot(d.t_s,d.cte_m*100,label='CTE [cm]')
    axes[3].set_ylabel('CTE [cm]');axes[3].set_xlabel('Time [s]')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.suptitle(scenario);fig.tight_layout()
    fig.savefig(folder/f'{scenario}_critic_value_vs_return.png',dpi=160);plt.close(fig)
    return {'critic_eval_'+key:value for key,value in summarize(d).items()}
