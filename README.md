# binance

基于 Python 和 Binance 历史行情的加密资产量化研究项目。使用 Codex CLI 将研究假设（thesis）转换成结构化方案和策略代码，再通过独立的回测与评估模块研究结果。

项目处于原型阶段。方案生成、代码生成、回测和评估目前分步运行，尚未接成自动研究循环。仓库中的策略和回测结果用于研究，不构成已经验证的交易系统。

## 当前流程

```text
thesis
  → agent.py → plan.json
  → coder.py → generated_signal.py
  → 检查数据要求、代码和目标权重
  → backtesting.py → evaluation.py
```

| 模块 | 当前功能 |
| --- | --- |
| `download_binance.py` | 下载 Binance 公共历史 K 线，处理重试、分页、去重和 UTC 时间 |
| `agent.py` | 调用 Codex CLI，将 thesis 转换成英文策略方案，按 JSON Schema 保存 |
| `plan.schema.json` | 规定方案的四个字段：假设依据、因子公式、权重规则、验证方法 |
| `coder.py` | 读取方案，调用另一次 Codex CLI，将最终代码输出保存为 `generated_signal.py` |
| `signals.py` | 手写 Signal 接口和回测示例，目前每个币固定 10% 权重 |
| `generated_signal.py` | 当前生成的月度拟合岭回归策略样例，使用收盘价构造特征 |
| `backtesting.py` | 加载行情，按目标权重调仓，记录持仓、现金、成本和净值，绘制净值及回撤 |
| `evaluation.py` | 收益、风险、换手率、成本、分期收益和可选基准比较 |
| `tests/test_evaluation.py` | 手算指标、日线/小时线年化、跨年收益、换手率、基准及多空回测验证 |

## 目录结构

```text
binance/
├── agent.py
├── coder.py
├── plan.schema.json
├── plan.json                 # 当前生成的方案样例，每次生成会覆盖
├── generated_signal.py       # 当前生成的代码样例，每次生成会覆盖
├── signals.py
├── backtesting.py
├── evaluation.py
├── download_binance.py
├── main.py                   # 预留入口，尚未编排整个研究流程
├── pyproject.toml
├── uv.lock
├── tests/
└── data/                     # 已跟踪的公开行情快照，9 个币 × 2 种频率
```

`outputs/` 用于保存图表，已加入 `.gitignore`。

## 安装

需要 Python 3.12+ 和 [uv](https://docs.astral.sh/uv/)。在项目根目录执行：

```bash
uv sync --locked
```

依赖包括 NumPy、pandas、Matplotlib、requests 和 OpenAI SDK。当前两个 Agent 脚本通过 subprocess 调用 Codex CLI，未使用 OpenAI SDK 发起 API 请求。

生成方案或代码还需要安装 [Codex CLI](https://learn.chatgpt.com/docs/cli)，并完成登录：

```bash
codex login
codex login status
```

使用 ChatGPT 登录时，按账号可用的 Codex 订阅额度及限制使用；API key 登录单独按 API 计费。参见 [官方登录说明](https://learn.chatgpt.com/docs/auth)。运行数据加载、已有策略、回测和评估不需要模型服务。

## 分步使用

### 1. 输入研究假设，生成方案

```bash
uv run agent.py
```

在 `Enter your thesis:` 后输入假设，例如：

```text
Assets with strong returns over the past week may continue outperforming next week.
```

结果保存到 `plan.json`，包含 `hypothesis_and_rationale`、`factor_formula`、`portfolio_weighting_rules` 和 `validation_plan`。当前 Schema 中四个字段都是字符串。`agent.py` 使用相对路径，请从项目根目录运行。

### 2. 根据方案生成策略代码

```bash
uv run coder.py
```

结果保存到 `generated_signal.py`。Agent 的命令执行环境使用 read-only sandbox；最终消息通过 CLI 的 `--output-last-message` 写入指定文件。参见 [Codex 命令说明](https://learn.chatgpt.com/docs/developer-commands#codex-exec)。

当前 `coder.py` 还没有检查 CLI 返回状态、生成代码语法或权重行为。生成后应先检查文件，再接入回测；文件存在本身不代表本轮生成成功。

### 3. 运行手写示例

```bash
uv run signals.py
```

当前示例使用 9 个币、小时线、每日调仓，每个币配置 10%，目标现金比例为 10%。脚本打印前几行权重、绩效指标、年度收益，并显示净值及回撤图。

`backtesting.py` 也提供独立等权示例：

```bash
uv run backtesting.py
```

运行前检查研究池、时间范围、数据目录、K 线频率和预热长度。`backtesting.py` 的示例与下载脚本仍包含本机绝对路径，需要按自己的环境修改；`signals.py` 和生成示例从项目位置定位数据目录。

### 4. 在 Python 中连接模块

```python
from backtesting import load_data, plot_backtest, run_backtest
from evaluation import evaluate_backtest
from signals import Signal

# config 结构可参考 signals.py 的运行示例。
data = load_data(config)
config["weights"] = Signal().get_weight(data["close"])
result = run_backtest(config, data=data)
report = evaluate_backtest(result)

print(report["summary"].to_string())
print(report["yearly_returns"])
plot_backtest(result, save_path="outputs/backtest.png")
```

验证生成策略后，可将导入改为 `from generated_signal import Signal`。

## 权重与回测约定

- `Signal.get_weight(close)` 返回与 close 同索引、同币种列的 DataFrame。
- 每行对应当前 K 线开盘时执行的目标权重，只能使用之前已经结束的 K 线数据。
- 正权重做多，负权重做空，每行绝对权重之和不超过 1，所有权重必须有限。
- `load_data()` 返回开盘价、收盘价、时间步长、起止时间。预热数据可用于生成信号，交易从配置的 start 开始。
- `missing_data="warn"` 将缺失或无效价格保留为 NaN 并警告；`"raise"` 则报错。缺开盘价时暂停该币交易，估值沿用最近已知价格。
- 回测支持 1h / 1d K 线，以及首次建仓后不再调仓（rebalance=None）、按小时、日、周、月调仓。
- 手续费和滑点按单边成交额计入，单位为 bps。做空按负持仓与现金记账，尚未模拟借币费用、资金费率、保证金、盘中强平或流动性约束；净资产非正时报错。

## 当前生成样例的限制

仓库内的 `plan.json` 与 `generated_signal.py` 是一次生成样例。该策略要求至少 20 个有效资产、253 根连续有效收盘价用于特征，以及至少 504 个已有标签的训练日期，并在月末拟合。

当前数据只有 9 个币，因此该生成样例返回全零权重；其运行示例又配置了小时线，而方案按日线设计。它目前不能用于验证这份方案的策略表现。后续应将实际研究池、数据频率和历史长度同时传入 Agent，并在配置不匹配时明确报错。仓库保留此样例用于展示当前流程的边界。

## 绩效评估

`evaluate_backtest(result, risk_free_rate=0.0, benchmarks=None)` 返回：

| 字段 | 内容 |
| --- | --- |
| summary | 总收益、年化复合收益、波动率、Sharpe、最大回撤、Calmar、盈利周期比例、最好/最差周期、成交数、换手率、手续费和滑点 |
| returns / drawdown | 账户收益和回撤时间序列 |
| turnover | 每根 K 线的双边换手率 |
| monthly_returns / yearly_returns | 按自然月、自然年复合的收益，含部分期间 |
| comparison | 策略与可选基准的收益和风险指标对照 |

比率使用小数，例如 0.12 表示 12%。年化按 365 天计算，自动从规则间隔的净值序列识别频率。Sharpe 使用周期算术超额收益和样本标准差；零波动率或不足两个收益观察值时为 NaN，零回撤时 Calmar 为 NaN。盈利周期比例统计 K 线账户收益，不是平仓交易胜率。

双边换手率包含首次建仓，分母为上一期末净资产，首根用初始资金。成本相对初始资金的比率只描述支出规模，不代表有无成本回测的收益差。分期收益按 K 线结束前的 UTC 日期归属，例如 1 月 1 日零点结算的日线收益属于前一年 12 月。

基准比较需传入已覆盖相同净值时点的序列：

```python
report = evaluate_backtest(
    result,
    benchmarks={"BTC": btc_result["equity"], "equal_weight": eq_result["equity"]},
)
print(report["comparison"])
```

## 数据

当前研究池：BTCUSDT、ETHUSDT、BNBUSDT、SOLUSDT、XRPUSDT、DOGEUSDT、ADAUSDT、TRXUSDT、LINKUSDT。仓库包含各自的 1h 和 1d CSV，共 18 个文件。时间为 UTC，文件名为 `{SYMBOL}_{INTERVAL}.csv`。

重新下载前检查 `download_binance.py` 中的交易对、起止日期和输出目录：

```bash
uv run download_binance.py
```

使用 Binance 公共行情接口，不需要交易 API key；同名数据文件会被覆盖。当前研究池为固定币种列表，尚未实现历史时点的成分重建。

## 验证

```bash
uv run python -m unittest discover -s tests -v
```

现有 8 项测试覆盖绩效指标和与多空回测的连接，尚未完整覆盖生成策略、Agent 调用失败、未来数据检查或全部交易记账场景。

## 接下来

- 将实际数据能力和回测配置传入规划、代码生成 Agent。
- 检查 CLI 失败、代码语法、权重约束、全零权重和未来数据使用。
- 统一配置、路径与运行入口，连接生成、验证、回测和评估。
- 增加交易记账测试、基准组合和样本外验证。
- 按实验保存 thesis、方案、代码版本、数据版本、参数与结果。
- 在以上基础上建立自动研究循环。
