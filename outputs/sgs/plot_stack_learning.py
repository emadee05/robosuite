"""Plot raw saved Stack evaluation curves; no simulator or training imports.

Requires matplotlib in the plotting Python environment. Training environments
need not be changed. Writes PNG/PDF charts and a CSV of the plotted measurements.
"""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'warmup':'#64748b','uniform':'#0072B2','sgs':'#D55E00'}
STAGES = ('tabletop','lifted','transport','placement-approach')
LABELS = {'warmup':'Uniform warmup','uniform':'Uniform continuation','sgs':'SGS continuation (kappa=5)'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    root=args.run.resolve()
    output=args.output_dir or root/'learning_curves'
    output.mkdir(parents=True,exist_ok=True)
    data={phase:json.loads((root/phase/'results.json').read_text()) for phase in COLORS}
    kappa=json.loads((root/'settings.json').read_text())['sgs']['kappa']
    LABELS['sgs']=f'SGS continuation (kappa={kappa:g})'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','savefig.facecolor':'white'})
    rows=[]
    for phase,result in data.items():
        for e in result['evaluations']:
            rows.append(dict(phase=phase,mode='deterministic' if e['deterministic_policy'] else 'sampled',
                added_transitions=e['additional_training_transitions'],total_transitions=e['training_transitions'],
                stage='overall',successes=e['successes'],episodes=e['episode_count'],success_rate=e['success_rate'],
                mean_episode_reward=sum(s['mean_episode_reward']*s['episodes'] for s in e['by_stage'].values())/e['episode_count']))
            for stage,summary in e['by_stage'].items():
                rows.append(dict(rows[-1],stage=stage,successes=summary['successes'],episodes=summary['episodes'],
                    success_rate=summary['success_rate'],mean_episode_reward=summary['mean_episode_reward']))
    with (output/'evaluation_curves.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    fig.suptitle('Stack PPO: evaluation success and dense reward',fontsize=17,fontweight='bold')
    for column,phases in enumerate((('warmup',),('uniform','sgs'))):
        top,bottom=axes[0,column],axes[1,column]
        top.set_title('Fresh uniform training' if column==0 else 'Continuations from the same warmup policy')
        for phase in phases:
            for deterministic in (False,True):
                records=[e for e in data[phase]['evaluations'] if e['deterministic_policy']==deterministic]
                records.sort(key=lambda e:e['additional_training_transitions'])
                x=[e['additional_training_transitions']/1e6 for e in records]
                y=[100*e['success_rate'] for e in records]
                label=('Sampled actions' if not deterministic else 'Deterministic actions') if phase=='warmup' else \
                      ('Uniform' if phase=='uniform' else 'SGS')+(' sampled' if not deterministic else ' deterministic')
                top.plot(x,y,color=COLORS[phase],linestyle='--' if deterministic else '-',
                         marker='s' if deterministic else 'o',markersize=4,label=label,alpha=.85)
                if not deterministic:
                    reward=[sum(s['mean_episode_reward']*s['episodes'] for s in e['by_stage'].values())/e['episode_count'] for e in records]
                    bottom.plot(x,reward,color=COLORS[phase],marker='o',markersize=4,label=LABELS[phase])
        top.set_ylim(-.8,25);top.set_ylabel('Successful evaluation episodes (%)')
        top.legend(fontsize=8,loc='upper right',framealpha=.95)
        bottom.set_ylim(0,800);bottom.set_ylabel('Mean episode return (sampled actions)')
        bottom.set_title('Dense reward during evaluation')
        for ax in (top,bottom):
            ax.set_xlabel('Training transitions within this phase (millions)')
            ax.set_xlim(-.04,3.06);ax.grid(alpha=.2);ax.set_axisbelow(True)
    fig.get_layout_engine().set(rect=(0,.06,1,.95))
    fig.text(.5,.025,'Raw checkpoint measurements; one training seed pair; evaluation uses the training pool.\n'
             'Sampled evaluation: 64 episodes per regular checkpoint, 160 at phase end. Deterministic: 32 episodes.',
             ha='center',fontsize=9,color='#475569')
    for extension in ('png','pdf'):
        fig.savefig(output/f'overall_learning_curves.{extension}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    fig.suptitle('Stack success by starting stage — sampled actions',fontsize=17,fontweight='bold')
    warmup_steps=data['warmup']['training_transitions']
    for ax,stage in zip(axes.flat,STAGES):
        for phase in COLORS:
            records=[e for e in data[phase]['evaluations'] if not e['deterministic_policy']]
            x=[e['training_transitions']/1e6 for e in records]
            y=[100*e['by_stage'][stage]['success_rate'] for e in records]
            ax.plot(x,y,color=COLORS[phase],marker='o',markersize=4,label=LABELS[phase])
        ax.axvline(warmup_steps/1e6,color='#94a3b8',linestyle=':',linewidth=1)
        ax.set_title(stage.replace('-',' ').title())
        ax.set_xlabel('Cumulative policy training transitions (millions)')
        ax.set_ylabel('Successful evaluation episodes (%)')
        ax.set_ylim(-1.5,50);ax.set_xlim(-.06,6.08);ax.grid(alpha=.2);ax.set_axisbelow(True)
    axes[0,0].legend(fontsize=8,loc='upper right')
    fig.get_layout_engine().set(rect=(0,.065,1,.95))
    fig.text(.5,.025,'Dotted line: warmup ends and the two independent continuations begin. Raw curves, no smoothing.\n'
             'Each stage: 16 episodes per regular checkpoint, 40 at phase end. A zero estimate is not proof of zero probability.',
             ha='center',fontsize=9,color='#475569')
    for extension in ('png','pdf'):
        fig.savefig(output/f'stage_learning_curves.{extension}',dpi=170)
    plt.close(fig)
    print('Saved learning curves and CSV:',output)


if __name__=='__main__':
    main()
