# 规模曲线资料

n>=0，scale>0，y为0到1的运行比例，variance为已知独立测量噪声方差，不能从重复运行结果重新估计后冒充已知。曲线exp的自变量是k*n/scale；k_grid严格升序且全正，数学搜索域就是该有限网格。对每个k，参数(a,b)属于三角形0<=b<=a<=1。目标sum((y-a+b exp(-k n/scale))²/variance)。同分目标以配置tie_tolerance（缺省1e-10）判断，优先较小k，其后a、b字典序最小；非唯一目标需提示，不利用目标容差任意放大识别精度。

profile接受阈值以全域最佳目标+cutoff判断，不连续段仅按网格相邻关系标识；候选规模预测集合用整个目标次水平集与三角形的交，不只用每k最优点。增益是同一个(a,b,k)下f(candidate)-f(baseline)，上下界应直接对此差值求界，不能把两个独立预测界相减冒充联合增益。候选n>=baseline_n。新增采集费用在candidate>baseline时为fixed+per_sample×(candidate-baseline)，相等时为0。可以输出不确定性导致无法区分的候选，不自动排序为投资建议。

已有domain只按规模输出已知方差权重下的样本均值，不拟合饱和参数、目标集合或预测包络。
