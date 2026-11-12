# 昂贵采集投入的学习曲线轮廓拟合

这是现场机器人应用分析的合成资料项目。`data`保存业务记录，`analysis_data.io`负责严格JSON读取和原始字节指纹，`analysis_data.domain`提供已有的基础汇总，`analysis_data.curve`提供有界饱和曲线拟合、profile 可识别性诊断与联合预测/增益界。字段与数学口径见[资料约定](docs/data-contract.md)，拟合与求界细节见[饱和曲线轮廓说明](docs/saturation-profile.md)。所有厂区、设备、收益与结果均为合成，不代表任何实际项目的效果。

Python 3.11及以上版本可直接运行，当前无第三方Python依赖。项目根目录中的常用命令：

```sh
python -m unittest discover -s tests -v
python -m compileall -q analysis_data tests
python -m analysis_data.cli inspect data/normal.json
python -m analysis_data.cli profile data/normal.json
python -m analysis_data.cli crosscheck data/flat-platform.json --candidate-n 2500 --steps 200
```

`inspect`展示原始资料摘要与基础统计；`profile`在声明的 k 网格上逐点拟合，输出每个网格点的目标、约束残差、轮廓接受分段，以及候选新增量、费用、预测增益联合界；`crosscheck`用密集 (a,b) 穷举交叉检查解析包络，并对照“只连最佳曲线”的过窄区间。拟合只读取历史汇总行，不启动训练或调用任何学习模型。结果写入其他目录时应保留输入来源指纹，原始资料不覆盖。更多边界资料与normal采用同样的JSON结构，invalid前缀文件保存不合法输入。

容器可执行 `docker build -t field-analysis .` 后用 `docker run --rm field-analysis` 查看资料，或 `docker run --rm field-analysis python -m unittest discover -s tests -v` 运行现有检查。运行期间无外网或账号要求。
