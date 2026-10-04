# E6：独立训练种子 + 匹配推理种子的正式确认

附件 E6 要同时排除 hard-Gumbel 推理波动、模型初始化/训练波动与单次偶然性。因此 E6 默认使用 `10 11 12`，**每个 seed 从头训练一份 RGB baseline**，再从该 RGB 权重训练同编号的 UDR-v1；保持已实测选出的 E2 uncertainty 方法与 E3 局部 alpha 强度不变，但为该 seed 分别训练 E2、E3 来源模型，严格合并其权重，再训练一份 UDR-v2 (E4)。三条链路互不共用训练权重。测试时同编号的 RGB、UDR-v1、UDR-v2 使用同一个 seed，并复用原项目排序的五数据集、×4、uint8 Y-channel、crop 4 流程。

E6 不增加新架构或损失。生成的配置逐项继承 E0/E2/E3/E4 的数据路径、优化器、迭代预算：RGB 从头训练 500k，固定取历史口径 `net_g_490000.pth`；UDR-v1 A/B 各 100k；E2 U1/U2 只做 B 100k，U3 做 head-only A 30k + B 100k；E3 A 30k + B 100k；E4 A 30k + B 100k。每条链固定选中的 U/A *方法*，但独立生成并训练其全部权重。E4 的逐 seed 初始权重只从**相同 seed**的 E2/E3 完整 checkpoint 合并，先核对共享 tensor schema，再由生产 E4 模型严格加载。所有新增流程均有源码注释。

目前未提供 E2/E3 的真实报告和最终 E4 权重，因此仓库不会预设最佳 U/A 或填写 E6 性能。先按 [E2 指南](E2_GUIDE.md)、[E3 指南](E3_GUIDE.md)完成上游实验，再运行 [E4 选择器](E4_GUIDE.md)以固定变体。E6 的三个 seed 使用这**同一选择规则的胜出变体**，避免每个 seed 事后挑选最佳结构。

## 在 CUDA 服务器准备 E6

```bash
git clone https://github.com/KaiXu-HIT/v3.2-M3SR-MambaIRv2-E6.git
cd v3.2-M3SR-MambaIRv2-E6
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
E2ROOT=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E2
E3ROOT=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E3
E0STATS="$E2ROOT/results/E0_udr_mechanism_audit/E0_statistics.json"
RGB=/home/BRAIN/xukai/code/v1.0-M3SR-MambaIRv2/experiments/v1.0_RGB_MambaIRv2_x4/models/net_g_490000.pth

# 只用于固定 E2/E3 方法和生成 E4 模板；E6 不复用这些历史训练权重。
python scripts/udr/prepare_e4_integration.py \
  --e2-results "$E2ROOT/results/E2_udrv2_uncertainty" \
  --e3-results "$E3ROOT/results/E3_local_alpha" \
  --e0-statistics "$E0STATS" \
  --rgb-teacher-checkpoint "$RGB"

python scripts/udr/prepare_e6_multiseed.py --stage prepare --seeds 10 11 12
python scripts/udr/check_e6_multiseed.py
python scripts/udr/run_e6_training.py --dry-run
```

`experiments/E6_plan/E6_plan.json` 给出逐 seed 依赖与预期 checkpoint；`E6_commands.md` 写出每一条独立训练命令，供审查。确认后按顺序执行全部训练：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/udr/run_e6_training.py
```

执行器先训练该 seed 的 RGB、UDR-v1、E2、E3，然后严格合并其 E4 初始权重，再训练 E4 A/B；逐步检查预期 checkpoint，发现已有输出时默认停止，防止跨 seed 混用。若中断且确认已有输出属于本次计划，可运行：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/udr/run_e6_training.py --skip-completed
```

`--skip-completed` 对已有 E4 合并权重还会核对其 SHA256 来源记录；最终评估再次检查每个 seed 的 E2/E3 来源哈希及 RGB/UDR-v1/UDR-v2 权重在三个 seed 之间各不相同。不要跨阶段使用 `--auto_resume`，也不要把一份 E4 权重复制成三个种子。

## 正式测试及可选 E2/E3 对照

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/udr/evaluate_e6_multiseed.py \
  --plan experiments/E6_plan/E6_plan.json \
  --output results/E6_multiseed

# 如需同时列出选中的 E2-only / E3-only：
CUDA_VISIBLE_DEVICES=0 python scripts/udr/evaluate_e6_multiseed.py \
  --plan experiments/E6_plan/E6_plan.json \
  --include-e2-e3 \
  --output results/E6_multiseed_with_E2_E3
```

默认仅保留方案指定的 RGB baseline、UDR-v1、UDR-v2 final。每个 seed 的三模型在独立子进程中使用相同手动种子，测试脚本会检查 checkpoint 存在、哈希互异、E4 选择未变化、逐 seed 的 E2/E3→E4 合并来源和原五集路径/指标完全一致。`E6_multiseed.json` 含原始逐 seed PSNR/SSIM、逐数据集 mean±std、相对**同 seed RGB** 的 ΔPSNR mean±std、E4 相对 UDR-v1 的增益、五集等权平均与证据等级；`E6_multiseed.md` 是汇总表。

证据等级严格按方案的 3-seed 规则：3/3 ΔPSNR>0 为 strong；均值>0 且 2/3 positive 为 moderate；均值>0 但仅 1/3 positive 为 weak；其他不判正增益。若计算资源允许，配置 `--seeds 10 20 30 40 50` 后同样训练与测试，5-seed 版本把 moderate 明确延伸为至少 4/5 positive（并非附件原文新增结论）。这些种子值同时驱动独立训练和配对推理；报告的 seed 间标准差包含这两种波动，不能解释为纯推理随机性。
