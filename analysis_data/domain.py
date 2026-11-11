"""按规模查看已知方差加权的运行结果均值。"""
from collections import defaultdict
from .io import finite,unique
def summarize(value):
    unique(value['runs'],'run_id');groups=defaultdict(lambda:[0.,0.,0])
    if finite(value['scale'])<=0:raise ValueError('规模换算必须为正')
    for row in value['runs']:
        n=finite(row['n']);y=finite(row['y']);v=finite(row['variance'])
        if n<0 or not 0<=y<=1 or v<=0:raise ValueError('规模比例或测量方差错误')
        g=groups[n];g[0]+=y/v;g[1]+=1/v;g[2]+=1
    return [{'n':n,'weighted_mean':g[0]/g[1],'weight':g[1],'runs':g[2]} for n,g in sorted(groups.items())]
