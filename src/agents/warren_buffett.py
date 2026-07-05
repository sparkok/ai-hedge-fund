"""
沃伦·巴菲特智能体

"以合理价格买入优秀企业，而非以优秀价格买入平庸企业。"
—— 沃伦·巴菲特

本模块模拟巴菲特的投资决策流程：
1. 获取财务数据（ROE、负债率、利润率、护城河指标等）
2. 多维度量化打分（基础分 ~22 分）
3. 三阶段 DCF 估值模型计算内在价值
4. 打包分数给 LLM，让 LLM 扮演巴菲特做最终判断
"""

from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import json
from typing_extensions import Literal
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


# ──────────────────────────────────────────────────────────────────────────
# 信号数据模型
# ──────────────────────────────────────────────────────────────────────────

class WarrenBuffettSignal(BaseModel):
    """巴菲特分析师的输出信号格式"""
    signal: Literal["bullish", "bearish", "neutral"]  # 看涨/看跌/中性
    confidence: int = Field(description="Confidence 0-100")  # 置信度 0-100
    reasoning: str = Field(description="Reasoning for the decision")  # 决策理由


# ──────────────────────────────────────────────────────────────────────────
# 主入口函数
# ──────────────────────────────────────────────────────────────────────────

def warren_buffett_agent(state: AgentState, agent_id: str = "warren_buffett_agent"):
    """
    巴菲特分析师的主函数。
    
    流程：
    1. 遍历每只股票
    2. 获取财务数据（指标 + 行项目）
    3. 运行多个维度的分析（基本面、一致性、护城河、定价权、账面价值、管理层）
    4. 计算内在价值（DCF）
    5. 计算安全边际
    6. 把所有分析结果打包，让 LLM 扮演巴菲特输出最终信号
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    # 收集所有分析结果，后续传给 LLM
    analysis_data = {}
    buffett_analysis = {}

    for ticker in tickers:
        # ── 步骤1：获取财务指标（ROE、负债率、利润率等） ──
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        # 要求取10期数据以便做趋势分析
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=10, api_key=api_key)

        # ── 步骤2：获取财务报表行项目（收入、利润、负债等） ──
        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "capital_expenditure",              # 资本支出
                "depreciation_and_amortization",     # 折旧与摊销
                "net_income",                        # 净利润
                "outstanding_shares",                # 流通股数
                "total_assets",                      # 总资产
                "total_liabilities",                 # 总负债
                "shareholders_equity",               # 股东权益
                "dividends_and_other_cash_distributions",  # 分红
                "issuance_or_purchase_of_equity_shares",    # 股票发行/回购
                "gross_profit",                      # 毛利润
                "revenue",                           # 营业收入
                "free_cash_flow",                    # 自由现金流
            ],
            end_date,
            period="ttm",
            limit=10,
            api_key=api_key,
        )

        # ── 步骤3：获取市值 ──
        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        # ── 步骤4：运行各维度分析 ──
        progress.update_status(agent_id, ticker, "Analyzing fundamentals")
        fundamental_analysis = analyze_fundamentals(metrics)          # 基本面（ROE/负债/利润率）

        progress.update_status(agent_id, ticker, "Analyzing consistency")
        consistency_analysis = analyze_consistency(financial_line_items)  # 盈利持续性

        progress.update_status(agent_id, ticker, "Analyzing competitive moat")
        moat_analysis = analyze_moat(metrics)                        # 护城河

        progress.update_status(agent_id, ticker, "Analyzing pricing power")
        pricing_power_analysis = analyze_pricing_power(financial_line_items, metrics)  # 定价权

        progress.update_status(agent_id, ticker, "Analyzing book value growth")
        book_value_analysis = analyze_book_value_growth(financial_line_items)  # 账面价值增长

        progress.update_status(agent_id, ticker, "Analyzing management quality")
        mgmt_analysis = analyze_management_quality(financial_line_items)  # 管理层质量

        # ── 步骤5：计算内在价值（DCF） ──
        progress.update_status(agent_id, ticker, "Calculating intrinsic value")
        intrinsic_value_analysis = calculate_intrinsic_value(financial_line_items)

        # ── 步骤6：汇总总分（LLM 会处理能力圈判断，所以这里只加硬指标分） ──
        total_score = (
            fundamental_analysis["score"] +
            consistency_analysis["score"] +
            moat_analysis["score"] +
            mgmt_analysis["score"] +
            pricing_power_analysis["score"] +
            book_value_analysis["score"]
        )

        # 各维度满分汇总
        max_possible_score = (
            10 +                        # 基本面：ROE(2) + 负债率(2) + 利润率(2) + 流动比率(1) + 未列出的额外容错
            moat_analysis["max_score"] +  # 护城河：5分
            mgmt_analysis["max_score"] +  # 管理层：2分
            5 +                          # 定价权：5分
            5                            # 账面价值增长：5分
        )

        # ── 步骤7：计算安全边际（巴菲特的核心概念） ──
        # 安全边际 = (内在价值 - 市值) / 市值，正值表示股价低于内在价值
        margin_of_safety = None
        intrinsic_value = intrinsic_value_analysis["intrinsic_value"]
        if intrinsic_value and market_cap:
            margin_of_safety = (intrinsic_value - market_cap) / market_cap

        # ── 步骤8：打包所有分析结果 ──
        analysis_data[ticker] = {
            "ticker": ticker,
            "score": total_score,
            "max_score": max_possible_score,
            "fundamental_analysis": fundamental_analysis,
            "consistency_analysis": consistency_analysis,
            "moat_analysis": moat_analysis,
            "pricing_power_analysis": pricing_power_analysis,
            "book_value_analysis": book_value_analysis,
            "management_analysis": mgmt_analysis,
            "intrinsic_value_analysis": intrinsic_value_analysis,
            "market_cap": market_cap,
            "margin_of_safety": margin_of_safety,
        }

        # ── 步骤9：让 LLM 扮演巴菲特做最终判断 ──
        progress.update_status(agent_id, ticker, "Generating Warren Buffett analysis")
        buffett_output = generate_buffett_output(
            ticker=ticker,
            analysis_data=analysis_data[ticker],
            state=state,
            agent_id=agent_id,
        )

        # 以统一格式存储分析结果
        buffett_analysis[ticker] = {
            "signal": buffett_output.signal,
            "confidence": buffett_output.confidence,
            "reasoning": buffett_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=buffett_output.reasoning)

    # 创建消息，传给图的下一个节点
    message = HumanMessage(content=json.dumps(buffett_analysis), name=agent_id)

    # 如果开启了 show_reasoning，打印推理过程
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(buffett_analysis, agent_id)

    # 把信号存入 state，后续被投资组合经理使用
    state["data"]["analyst_signals"][agent_id] = buffett_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


# ══════════════════════════════════════════════════════════════════════════
# 以下为各维度分析函数
# ══════════════════════════════════════════════════════════════════════════


def analyze_fundamentals(metrics: list) -> dict[str, any]:
    """
    分析公司基本面（基于巴菲特的选股标准）。

    考核指标：
    - ROE（净资产收益率）：> 15%，巴菲特偏好持续高ROE
    - 负债/权益比：< 0.5，巴菲特厌恶高负债
    - 营业利润率：> 15%，好生意应该有高利润率
    - 流动比率：> 1.5，良好的流动性避免危机
    
    满分约 7 分（ROE 2 + 负债率 2 + 利润率 2 + 流动比率 1）
    """
    if not metrics:
        return {"score": 0, "details": "Insufficient fundamental data"}

    latest_metrics = metrics[0]

    score = 0
    reasoning = []

    # ── ROE（净资产收益率）：巴菲特最看重的指标之一 ──
    # ROE > 15% 说明公司能有效利用股东资本创造利润
    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:
        score += 2
        reasoning.append(f"Strong ROE of {latest_metrics.return_on_equity:.1%}")
    elif latest_metrics.return_on_equity:
        reasoning.append(f"Weak ROE of {latest_metrics.return_on_equity:.1%}")
    else:
        reasoning.append("ROE data not available")

    # ── 负债/权益比：巴菲特喜欢低负债公司 ──
    # 低于 0.5 表示公司负债水平保守，抗风险能力强
    if latest_metrics.debt_to_equity and latest_metrics.debt_to_equity < 0.5:
        score += 2
        reasoning.append("Conservative debt levels")
    elif latest_metrics.debt_to_equity:
        reasoning.append(f"High debt to equity ratio of {latest_metrics.debt_to_equity:.1f}")
    else:
        reasoning.append("Debt to equity data not available")

    # ── 营业利润率：好生意应该能赚钱 ──
    # > 15% 说明公司有竞争优势，能收取溢价
    if latest_metrics.operating_margin and latest_metrics.operating_margin > 0.15:
        score += 2
        reasoning.append("Strong operating margins")
    elif latest_metrics.operating_margin:
        reasoning.append(f"Weak operating margin of {latest_metrics.operating_margin:.1%}")
    else:
        reasoning.append("Operating margin data not available")

    # ── 流动比率：短期偿债能力 ──
    # > 1.5 说明流动资产覆盖流动负债有余，财务稳健
    if latest_metrics.current_ratio and latest_metrics.current_ratio > 1.5:
        score += 1
        reasoning.append("Good liquidity position")
    elif latest_metrics.current_ratio:
        reasoning.append(f"Weak liquidity with current ratio of {latest_metrics.current_ratio:.1f}")
    else:
        reasoning.append("Current ratio data not available")

    return {"score": score, "details": "; ".join(reasoning), "metrics": latest_metrics.model_dump()}


def analyze_consistency(financial_line_items: list) -> dict[str, any]:
    """
    分析盈利持续性。

    巴菲特喜欢"可预测"的公司——那些年复一年稳定增长利润的企业。
    这里检查：
    - 净利润是否连续多期持续增长
    - 整体增长率
    """
    if len(financial_line_items) < 4:  # 至少需要4期数据才能做趋势分析
        return {"score": 0, "details": "Insufficient historical data"}

    score = 0
    reasoning = []

    # 提取各期净利润
    earnings_values = [item.net_income for item in financial_line_items if item.net_income]
    if len(earnings_values) >= 4:
        # 检查是否每一期都比上一期增长
        earnings_growth = all(earnings_values[i] > earnings_values[i + 1] for i in range(len(earnings_values) - 1))

        if earnings_growth:
            score += 3
            reasoning.append("Consistent earnings growth over past periods")
        else:
            reasoning.append("Inconsistent earnings growth pattern")

        # 计算整个期间的增长率
        if len(earnings_values) >= 2 and earnings_values[-1] != 0:
            growth_rate = (earnings_values[0] - earnings_values[-1]) / abs(earnings_values[-1])
            reasoning.append(f"Total earnings growth of {growth_rate:.1%} over past {len(earnings_values)} periods")
    else:
        reasoning.append("Insufficient earnings data for trend analysis")

    return {
        "score": score,
        "details": "; ".join(reasoning),
    }


def analyze_moat(metrics: list) -> dict[str, any]:
    """
    评估公司的护城河（持久竞争优势）。

    护城河是巴菲特投资哲学的核心。他寻找的是"有宽护城河"的公司——
    那些竞争对手难以复制的竞争优势。

    这里从4个维度评估：
    1. ROE 持续性：长期高 ROE 表明有竞争壁垒
    2. 利润率稳定性：稳定的高利润率 = 定价权
    3. 资产效率：高效的资产运用 = 运营优势
    4. 综合稳定性：整体业绩波动小 = 竞争地位强

    满分：5分
    """
    if not metrics or len(metrics) < 5:  # 需要5期以上数据
        return {"score": 0, "max_score": 5, "details": "Insufficient data for comprehensive moat analysis"}

    reasoning = []
    moat_score = 0
    max_score = 5

    # ── 维度1：ROE 持续性（巴菲特最看重的护城河指标） ──
    # 如果公司能长期维持 > 15% 的 ROE，说明有真正的竞争优势
    historical_roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
    historical_roics = [m.return_on_invested_capital for m in metrics if
                        hasattr(m, 'return_on_invested_capital') and m.return_on_invested_capital is not None]

    if len(historical_roes) >= 5:
        high_roe_periods = sum(1 for roe in historical_roes if roe > 0.15)
        roe_consistency = high_roe_periods / len(historical_roes)

        if roe_consistency >= 0.8:  # 80%+ 的期数 ROE > 15%
            moat_score += 2
            avg_roe = sum(historical_roes) / len(historical_roes)
            reasoning.append(
                f"Excellent ROE consistency: {high_roe_periods}/{len(historical_roes)} periods >15% (avg: {avg_roe:.1%}) - indicates durable competitive advantage")
        elif roe_consistency >= 0.6:
            moat_score += 1
            reasoning.append(f"Good ROE performance: {high_roe_periods}/{len(historical_roes)} periods >15%")
        else:
            reasoning.append(f"Inconsistent ROE: only {high_roe_periods}/{len(historical_roes)} periods >15%")
    else:
        reasoning.append("Insufficient ROE history for moat analysis")

    # ── 维度2：营业利润率稳定性（定价能力指标） ──
    # 稳定的利润率意味着公司能抵御竞争和通胀
    historical_margins = [m.operating_margin for m in metrics if m.operating_margin is not None]
    if len(historical_margins) >= 5:
        avg_margin = sum(historical_margins) / len(historical_margins)
        recent_margins = historical_margins[:3]   # 最近3期
        older_margins = historical_margins[-3:]   # 最早3期

        recent_avg = sum(recent_margins) / len(recent_margins)
        older_avg = sum(older_margins) / len(older_margins)

        if avg_margin > 0.2 and recent_avg >= older_avg:  # 20%+ 利润率且稳定/改善
            moat_score += 1
            reasoning.append(f"Strong and stable operating margins (avg: {avg_margin:.1%}) indicate pricing power moat")
        elif avg_margin > 0.15:  # 至少还不错的利润率
            reasoning.append(f"Decent operating margins (avg: {avg_margin:.1%}) suggest some competitive advantage")
        else:
            reasoning.append(f"Low operating margins (avg: {avg_margin:.1%}) suggest limited pricing power")

    # ── 维度3：资产效率与规模优势 ──
    if len(metrics) >= 5:
        asset_turnovers = []
        for m in metrics:
            if hasattr(m, 'asset_turnover') and m.asset_turnover is not None:
                asset_turnovers.append(m.asset_turnover)

        if len(asset_turnovers) >= 3:
            if any(turnover > 1.0 for turnover in asset_turnovers):  # 资产周转率 > 1 表示高效
                moat_score += 1
                reasoning.append("Efficient asset utilization suggests operational moat")

    # ── 维度4：竞争地位强度（通过稳定性推断） ──
    # 计算 ROE 和利润率的变异系数，越低越稳定
    if len(historical_roes) >= 5 and len(historical_margins) >= 5:
        roe_avg = sum(historical_roes) / len(historical_roes)
        roe_variance = sum((roe - roe_avg) ** 2 for roe in historical_roes) / len(historical_roes)
        roe_stability = 1 - (roe_variance ** 0.5) / roe_avg if roe_avg > 0 else 0

        margin_avg = sum(historical_margins) / len(historical_margins)
        margin_variance = sum((margin - margin_avg) ** 2 for margin in historical_margins) / len(historical_margins)
        margin_stability = 1 - (margin_variance ** 0.5) / margin_avg if margin_avg > 0 else 0

        overall_stability = (roe_stability + margin_stability) / 2

        if overall_stability > 0.7:  # 高稳定性 = 强竞争地位
            moat_score += 1
            reasoning.append(f"High performance stability ({overall_stability:.1%}) suggests strong competitive moat")

    # 封顶
    moat_score = min(moat_score, max_score)

    return {
        "score": moat_score,
        "max_score": max_score,
        "details": "; ".join(reasoning) if reasoning else "Limited moat analysis available",
    }


def analyze_management_quality(financial_line_items: list) -> dict[str, any]:
    """
    分析管理层质量。

    巴菲特非常看重管理层是否"善待股东"：
    - 回购股票（表明管理层认为股票被低估）
    - 分红（持续的现金回报）
    - 不增发稀释（不损害现有股东利益）

    满分：2分
    """
    if not financial_line_items:
        return {"score": 0, "max_score": 2, "details": "Insufficient data for management analysis"}

    reasoning = []
    mgmt_score = 0

    latest = financial_line_items[0]

    # ── 检查股票回购 ──
    # issuance_or_purchase_of_equity_shares < 0 表示公司花钱回购股票
    if hasattr(latest,
               "issuance_or_purchase_of_equity_shares") and latest.issuance_or_purchase_of_equity_shares and latest.issuance_or_purchase_of_equity_shares < 0:
        mgmt_score += 1
        reasoning.append("Company has been repurchasing shares (shareholder-friendly)")

    # ── 检查是否增发 ──
    # 正值表示发行新股，可能稀释现有股东
    if hasattr(latest,
               "issuance_or_purchase_of_equity_shares") and latest.issuance_or_purchase_of_equity_shares and latest.issuance_or_purchase_of_equity_shares > 0:
        reasoning.append("Recent common stock issuance (potential dilution)")
    else:
        reasoning.append("No significant new stock issuance detected")

    # ── 检查分红记录 ──
    # 持续分红说明公司有稳定的现金流和股东回报文化
    if hasattr(latest,
               "dividends_and_other_cash_distributions") and latest.dividends_and_other_cash_distributions and latest.dividends_and_other_cash_distributions < 0:
        mgmt_score += 1
        reasoning.append("Company has a track record of paying dividends")
    else:
        reasoning.append("No or minimal dividends paid")

    return {
        "score": mgmt_score,
        "max_score": 2,
        "details": "; ".join(reasoning),
    }


# ══════════════════════════════════════════════════════════════════════════
# 内在价值计算相关函数
# ══════════════════════════════════════════════════════════════════════════


def calculate_owner_earnings(financial_line_items: list) -> dict[str, any]:
    """
    计算所有者收益（Owner Earnings）—— 巴菲特偏好的真实盈利能力度量。

    公式：
    所有者收益 = 净利润 + 折旧摊销 - 维护性资本支出 - 营运资本变动

    巴菲特认为 GAAP 净利润不能反映真实盈利能力，因为：
    - 净利润扣除了折旧（非现金支出），但公司需要花钱维持设备
    - 资本支出中只有"维护性"部分才是真正的成本，"增长性"支出是投资
    """
    if not financial_line_items or len(financial_line_items) < 2:
        return {"owner_earnings": None, "details": ["Insufficient data for owner earnings calculation"]}

    latest = financial_line_items[0]
    details = []

    # 核心要素
    net_income = latest.net_income                    # 净利润
    depreciation = latest.depreciation_and_amortization  # 折旧摊销（非现金支出，所以要加回）
    capex = latest.capital_expenditure               # 资本支出

    if not all([net_income is not None, depreciation is not None, capex is not None]):
        missing = []
        if net_income is None: missing.append("net income")
        if depreciation is None: missing.append("depreciation")
        if capex is None: missing.append("capital expenditure")
        return {"owner_earnings": None, "details": [f"Missing components: {', '.join(missing)}"]}

    # 用历史数据估算维护性资本支出
    maintenance_capex = estimate_maintenance_capex(financial_line_items)

    # 营运资本变动分析
    working_capital_change = 0
    if len(financial_line_items) >= 2:
        try:
            current_assets_current = getattr(latest, 'current_assets', None)
            current_liab_current = getattr(latest, 'current_liabilities', None)

            previous = financial_line_items[1]
            current_assets_previous = getattr(previous, 'current_assets', None)
            current_liab_previous = getattr(previous, 'current_liabilities', None)

            if all([current_assets_current, current_liab_current, current_assets_previous, current_liab_previous]):
                wc_current = current_assets_current - current_liab_current
                wc_previous = current_assets_previous - current_liab_previous
                working_capital_change = wc_current - wc_previous
                details.append(f"Working capital change: ${working_capital_change:,.0f}")
        except:
            pass  # 数据不足时跳过营运资本调整

    # 计算所有者收益
    owner_earnings = net_income + depreciation - maintenance_capex - working_capital_change

    # 合理性检查
    if owner_earnings < net_income * 0.3:  # 如果所有者收益不到净利润30%，说明是重资产公司
        details.append("Warning: Owner earnings significantly below net income - high capex intensity")

    if maintenance_capex > depreciation * 2:  # 维护性支出超过折旧2倍，值得警告
        details.append("Warning: Estimated maintenance capex seems high relative to depreciation")

    details.extend([
        f"Net income: ${net_income:,.0f}",
        f"Depreciation: ${depreciation:,.0f}",
        f"Estimated maintenance capex: ${maintenance_capex:,.0f}",
        f"Owner earnings: ${owner_earnings:,.0f}"
    ])

    return {
        "owner_earnings": owner_earnings,
        "components": {
            "net_income": net_income,
            "depreciation": depreciation,
            "maintenance_capex": maintenance_capex,
            "working_capital_change": working_capital_change,
            "total_capex": abs(capex) if capex else 0
        },
        "details": details,
    }


def estimate_maintenance_capex(financial_line_items: list) -> float:
    """
    估算维护性资本支出。

    难点：财报中的资本支出（Capex）包含"维护性"和"增长性"两部分。
    巴菲特认为只有维护性部分才是真正的成本。
    
    这里用三种方法估算，取中位数：
    1. 总 Capex 的 85%（假设15%是增长性支出）
    2. 等于折旧额（维持现有资产）
    3. 历史 Capex/收入 比率 × 当前收入
    """
    if not financial_line_items:
        return 0

    # 方法1：历史 Capex 占收入的比例
    capex_ratios = []
    depreciation_values = []

    for item in financial_line_items[:5]:  # 最近5期
        if hasattr(item, 'capital_expenditure') and hasattr(item, 'revenue'):
            if item.capital_expenditure and item.revenue and item.revenue > 0:
                capex_ratio = abs(item.capital_expenditure) / item.revenue
                capex_ratios.append(capex_ratio)

        if hasattr(item, 'depreciation_and_amortization') and item.depreciation_and_amortization:
            depreciation_values.append(item.depreciation_and_amortization)

    # 方法2：折旧的百分比（维护性支出通常为折旧的80-120%）
    latest_depreciation = financial_line_items[0].depreciation_and_amortization if financial_line_items[
        0].depreciation_and_amortization else 0

    # 当前总 Capex
    latest_capex = abs(financial_line_items[0].capital_expenditure) if financial_line_items[
        0].capital_expenditure else 0

    # 保守估算：取三者中的中位数
    method_1 = latest_capex * 0.85  # 总Capex的85%
    method_2 = latest_depreciation   # 100%折旧

    if len(capex_ratios) >= 3:
        avg_capex_ratio = sum(capex_ratios) / len(capex_ratios)
        latest_revenue = financial_line_items[0].revenue if hasattr(financial_line_items[0], 'revenue') and \
                                                            financial_line_items[0].revenue else 0
        method_3 = avg_capex_ratio * latest_revenue if latest_revenue else 0

        # 取中位数（最保守）
        estimates = sorted([method_1, method_2, method_3])
        return estimates[1]  # 中位数
    else:
        # 取较高值（更保守）
        return max(method_1, method_2)


def calculate_intrinsic_value(financial_line_items: list) -> dict[str, any]:
    """
    计算内在价值 —— 使用三阶段 DCF 模型。

    巴菲特："内在价值是公司剩余生命周期内所有现金流的折现值。"
    
    模型结构：
    - 阶段1（5年）：较高增长期（基于历史增长，上限8%）
    - 阶段2（5年）：过渡期（阶段1的一半，上限4%）
    - 终值：永续增长 2.5%（长期GDP增长率）
    - 折现率：10%（巴菲特式保守）
    - 再打 85 折（额外的安全边际）
    """
    if not financial_line_items or len(financial_line_items) < 3:
        return {"intrinsic_value": None, "details": ["Insufficient data for reliable valuation"]}

    # 先算所有者收益
    earnings_data = calculate_owner_earnings(financial_line_items)
    if not earnings_data["owner_earnings"]:
        return {"intrinsic_value": None, "details": earnings_data["details"]}

    owner_earnings = earnings_data["owner_earnings"]
    latest_financial_line_items = financial_line_items[0]
    shares_outstanding = latest_financial_line_items.outstanding_shares

    if not shares_outstanding or shares_outstanding <= 0:
        return {"intrinsic_value": None, "details": ["Missing or invalid shares outstanding data"]}

    details = []

    # ── 估算增长率（基于历史表现，非常保守） ──
    historical_earnings = []
    for item in financial_line_items[:5]:  # 最近5年
        if hasattr(item, 'net_income') and item.net_income:
            historical_earnings.append(item.net_income)

    if len(historical_earnings) >= 3:
        oldest_earnings = historical_earnings[-1]
        latest_earnings = historical_earnings[0]
        years = len(historical_earnings) - 1

        if oldest_earnings > 0:
            historical_growth = ((latest_earnings / oldest_earnings) ** (1 / years)) - 1
            # 保守调整：限制范围并打折
            historical_growth = max(-0.05, min(historical_growth, 0.15))  # 限制在 -5% 到 15%
            conservative_growth = historical_growth * 0.7  # 打7折
        else:
            conservative_growth = 0.03  # 如果基期为负，默认3%
    else:
        conservative_growth = 0.03  # 数据不足时默认3%

    # ── 三阶段增长假设（巴菲特式保守） ──
    stage1_growth = min(conservative_growth, 0.08)       # 阶段1：上限8%
    stage2_growth = min(conservative_growth * 0.5, 0.04)  # 阶段2：阶段1的一半，上限4%
    terminal_growth = 0.025                               # 永续增长：2.5%（长期GDP）

    # 折现率：使用保守的10%
    discount_rate = 0.10

    # ── 三阶段 DCF 计算 ──
    stage1_years = 5  # 高增长阶段年数
    stage2_years = 5  # 过渡阶段年数

    details.append(
        f"Using three-stage DCF: Stage 1 ({stage1_growth:.1%}, {stage1_years}y), Stage 2 ({stage2_growth:.1%}, {stage2_years}y), Terminal ({terminal_growth:.1%})")

    # 阶段1：较高增长
    stage1_pv = 0
    for year in range(1, stage1_years + 1):
        future_earnings = owner_earnings * (1 + stage1_growth) ** year
        pv = future_earnings / (1 + discount_rate) ** year
        stage1_pv += pv

    # 阶段2：过渡增长
    stage2_pv = 0
    stage1_final_earnings = owner_earnings * (1 + stage1_growth) ** stage1_years
    for year in range(1, stage2_years + 1):
        future_earnings = stage1_final_earnings * (1 + stage2_growth) ** year
        pv = future_earnings / (1 + discount_rate) ** (stage1_years + year)
        stage2_pv += pv

    # 终值（Gordon Growth Model）
    final_earnings = stage1_final_earnings * (1 + stage2_growth) ** stage2_years
    terminal_earnings = final_earnings * (1 + terminal_growth)
    terminal_value = terminal_earnings / (discount_rate - terminal_growth)
    terminal_pv = terminal_value / (1 + discount_rate) ** (stage1_years + stage2_years)

    # 总内在价值
    intrinsic_value = stage1_pv + stage2_pv + terminal_pv

    # 额外的安全边际：再打85折（巴菲特式保守主义）
    conservative_intrinsic_value = intrinsic_value * 0.85

    details.extend([
        f"Stage 1 PV: ${stage1_pv:,.0f}",
        f"Stage 2 PV: ${stage2_pv:,.0f}",
        f"Terminal PV: ${terminal_pv:,.0f}",
        f"Total IV: ${intrinsic_value:,.0f}",
        f"Conservative IV (15% haircut): ${conservative_intrinsic_value:,.0f}",
        f"Owner earnings: ${owner_earnings:,.0f}",
        f"Discount rate: {discount_rate:.1%}"
    ])

    return {
        "intrinsic_value": conservative_intrinsic_value,
        "raw_intrinsic_value": intrinsic_value,
        "owner_earnings": owner_earnings,
        "assumptions": {
            "stage1_growth": stage1_growth,
            "stage2_growth": stage2_growth,
            "terminal_growth": terminal_growth,
            "discount_rate": discount_rate,
            "stage1_years": stage1_years,
            "stage2_years": stage2_years,
            "historical_growth": conservative_growth if 'conservative_growth' in locals() else None,
        },
        "details": details,
    }


def analyze_book_value_growth(financial_line_items: list) -> dict[str, any]:
    """
    分析每股账面价值增长。

    巴菲特早期投资策略的核心就是寻找"以低于账面价值交易"的股票。
    虽然后来他转向"以合理价格买入优质企业"，但账面价值增长仍然是
    衡量公司价值创造的重要指标。

    满分：5分（增长持续性 3分 + CAGR 2分）
    """
    if len(financial_line_items) < 3:
        return {"score": 0, "details": "Insufficient data for book value analysis"}

    # 计算每股账面价值序列
    book_values = [
        item.shareholders_equity / item.outstanding_shares
        for item in financial_line_items
        if hasattr(item, 'shareholders_equity') and hasattr(item, 'outstanding_shares')
        and item.shareholders_equity and item.outstanding_shares
    ]

    if len(book_values) < 3:
        return {"score": 0, "details": "Insufficient book value data for growth analysis"}

    score = 0
    reasoning = []

    # ── 增长稳定性评分（满分3分） ──
    growth_periods = sum(1 for i in range(len(book_values) - 1) if book_values[i] > book_values[i + 1])
    growth_rate = growth_periods / (len(book_values) - 1)

    if growth_rate >= 0.8:
        score += 3
        reasoning.append("Consistent book value per share growth (Buffett's favorite metric)")
    elif growth_rate >= 0.6:
        score += 2
        reasoning.append("Good book value per share growth pattern")
    elif growth_rate >= 0.4:
        score += 1
        reasoning.append("Moderate book value per share growth")
    else:
        reasoning.append("Inconsistent book value per share growth")

    # ── CAGR 评分（满分2分） ──
    cagr_score, cagr_reason = _calculate_book_value_cagr(book_values)
    score += cagr_score
    reasoning.append(cagr_reason)

    return {"score": score, "details": "; ".join(reasoning)}


def _calculate_book_value_cagr(book_values: list) -> tuple[int, str]:
    """
    计算每股账面价值的复合年增长率（CAGR）。
    
    处理各种边界情况（负值、从负转正等）。
    """
    if len(book_values) < 2:
        return 0, "Insufficient data for CAGR calculation"

    oldest_bv, latest_bv = book_values[-1], book_values[0]
    years = len(book_values) - 1

    if oldest_bv > 0 and latest_bv > 0:
        cagr = ((latest_bv / oldest_bv) ** (1 / years)) - 1
        if cagr > 0.15:
            return 2, f"Excellent book value CAGR: {cagr:.1%}"
        elif cagr > 0.1:
            return 1, f"Good book value CAGR: {cagr:.1%}"
        else:
            return 0, f"Book value CAGR: {cagr:.1%}"
    elif oldest_bv < 0 < latest_bv:
        return 3, "Excellent: Company improved from negative to positive book value"
    elif oldest_bv > 0 > latest_bv:
        return 0, "Warning: Company declined from positive to negative book value"
    else:
        return 0, "Unable to calculate meaningful book value CAGR due to negative values"


def analyze_pricing_power(financial_line_items: list, metrics: list) -> dict[str, any]:
    """
    分析定价权 —— 巴菲特眼中护城河的关键指标。

    定价权 = 公司能在不失去客户的情况下提价的能力。
    表现为毛利润率稳定或扩大。

    满分：5分
    - 毛利率趋势（改善/稳定/恶化）：3分
    - 毛利率绝对值（高/中/低）：2分
    """
    if not financial_line_items or not metrics:
        return {"score": 0, "details": "Insufficient data for pricing power analysis"}

    score = 0
    reasoning = []

    # ── 毛利率趋势分析 ──
    gross_margins = []
    for item in financial_line_items:
        if hasattr(item, 'gross_margin') and item.gross_margin is not None:
            gross_margins.append(item.gross_margin)

    if len(gross_margins) >= 3:
        recent_avg = sum(gross_margins[:2]) / 2 if len(gross_margins) >= 2 else gross_margins[0]
        older_avg = sum(gross_margins[-2:]) / 2 if len(gross_margins) >= 2 else gross_margins[-1]

        if recent_avg > older_avg + 0.02:  # 改善2%以上
            score += 3
            reasoning.append("Expanding gross margins indicate strong pricing power")
        elif recent_avg > older_avg:  # 轻微改善
            score += 2
            reasoning.append("Improving gross margins suggest good pricing power")
        elif abs(recent_avg - older_avg) < 0.01:  # 稳定（波动 < 1%）
            score += 1
            reasoning.append("Stable gross margins during economic uncertainty")
        else:
            reasoning.append("Declining gross margins may indicate pricing pressure")

    # ── 毛利率绝对水平 ──
    if gross_margins:
        avg_margin = sum(gross_margins) / len(gross_margins)
        if avg_margin > 0.5:  # > 50%：非常强
            score += 2
            reasoning.append(f"Consistently high gross margins ({avg_margin:.1%}) indicate strong pricing power")
        elif avg_margin > 0.3:  # > 30%：还不错
            score += 1
            reasoning.append(f"Good gross margins ({avg_margin:.1%}) suggest decent pricing power")

    return {
        "score": score,
        "details": "; ".join(reasoning) if reasoning else "Limited pricing power analysis available"
    }


# ══════════════════════════════════════════════════════════════════════════
# LLM 决策生成
# ══════════════════════════════════════════════════════════════════════════


def generate_buffett_output(
        ticker: str,
        analysis_data: dict[str, any],
        state: AgentState,
        agent_id: str = "warren_buffett_agent",
) -> WarrenBuffettSignal:
    """
    让 LLM 扮演巴菲特做出最终投资决策。

    前面的代码完成了量化打分 + DCF 估值，这里把结果打包发给 LLM。
    LLM 扮演巴菲特，按照他的决策框架判断：
    
    看涨 = 好生意 + 安全边际 > 0
    看跌 = 差生意或明显高估
    中性 = 好生意但价格不够便宜，或信号矛盾

    Prompt 设计原则：只依赖提供的 facts，不编造数据，返回 JSON。
    """

    # ── 构建简洁的事实摘要 ──
    # 把所有分析结果压缩成 LLM 能高效处理的格式
    facts = {
        "score": analysis_data.get("score"),                                         # 量化总分
        "max_score": analysis_data.get("max_score"),                                 # 满分
        "fundamentals": analysis_data.get("fundamental_analysis", {}).get("details"),  # 基本面分析详情
        "consistency": analysis_data.get("consistency_analysis", {}).get("details"),  # 盈利一致性
        "moat": analysis_data.get("moat_analysis", {}).get("details"),               # 护城河分析
        "pricing_power": analysis_data.get("pricing_power_analysis", {}).get("details"),  # 定价权
        "book_value": analysis_data.get("book_value_analysis", {}).get("details"),   # 账面价值
        "management": analysis_data.get("management_analysis", {}).get("details"),   # 管理层质量
        "intrinsic_value": analysis_data.get("intrinsic_value_analysis", {}).get("intrinsic_value"),  # 内在价值
        "market_cap": analysis_data.get("market_cap"),                               # 当前市值
        "margin_of_safety": analysis_data.get("margin_of_safety"),                    # 安全边际
    }

    # ── 构造提示词 ──
    # System 消息定义巴菲特的身份和决策规则
    # Human 消息传入具体股票和分析数据
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Warren Buffett. Decide bullish, bearish, or neutral using only the provided facts.\n"
                "\n"
                "Checklist for decision:\n"
                "- Circle of competence\n"
                "- Competitive moat\n"
                "- Management quality\n"
                "- Financial strength\n"
                "- Valuation vs intrinsic value\n"
                "- Long-term prospects\n"
                "\n"
                "Signal rules:\n"
                "- Bullish: strong business AND margin_of_safety > 0.\n"
                "- Bearish: poor business OR clearly overvalued.\n"
                "- Neutral: good business but margin_of_safety <= 0, or mixed evidence.\n"
                "\n"
                "Confidence scale:\n"
                "- 90-100%: Exceptional business within my circle, trading at attractive price\n"
                "- 70-89%: Good business with decent moat, fair valuation\n"
                "- 50-69%: Mixed signals, would need more information or better price\n"
                "- 30-49%: Outside my expertise or concerning fundamentals\n"
                "- 10-29%: Poor business or significantly overvalued\n"
                "\n"
                "Keep reasoning under 120 characters. Do not invent data. Return JSON only."
            ),
            (
                "human",
                "Ticker: {ticker}\n"
                "Facts:\n{facts}\n\n"
                "Return exactly:\n"
                "{{\n"
                '  "signal": "bullish" | "bearish" | "neutral",\n'
                '  "confidence": int,\n'
                '  "reasoning": "short justification"\n'
                "}}"
            ),
        ]
    )

    prompt = template.invoke({
        "facts": json.dumps(facts, separators=(",", ":"), ensure_ascii=False),
        "ticker": ticker,
    })

    # 后备方案：如果 LLM 调用失败，返回中性信号
    def create_default_warren_buffett_signal():
        return WarrenBuffettSignal(signal="neutral", confidence=50, reasoning="Insufficient data")

    return call_llm(
        prompt=prompt,
        pydantic_model=WarrenBuffettSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_warren_buffett_signal,
    )
