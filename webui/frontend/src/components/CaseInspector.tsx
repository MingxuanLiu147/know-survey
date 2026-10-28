import { ChevronLeft, ChevronRight, FileQuestion, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { ApiError } from "../api";
import { displayTechnicalText, resultReasonText, uiErrorMessage, useI18n } from "../i18n";
import type { ResultSummary } from "../types";
import { Button, EmptyState, InlineError, LoadingBlock } from "./ui";
import { metricDisplayLabel } from "./MetricsView";

interface CaseRecord {
  id: string;
  metric_id: string;
  evaluator: string | null;
  phase: string | null;
  case_index: string | number | null;
  probe_index: number | null;
  group: string | null;
  prompt: string | null;
  expected: string | null;
  before: string | null;
  after: string | null;
  status: string;
  measurement_kind: string | null;
  score: number | null;
  before_score: number | null;
  after_score: number | null;
  source: string;
  source_pointer: string;
  unavailable_reason: string | null;
  warnings: string[];
  protocol_version: string | null;
  scoring: string | null;
}

interface CasePage {
  result_id: string;
  offset: number;
  limit: number;
  total: number;
  has_more: boolean;
  cases: CaseRecord[];
  warnings: string[];
}

/** Read saved per-example outputs without substituting reference answers. */
export function CaseInspector({ results, metricId }: { results: ResultSummary[]; metricId?: string }) {
  const { language, t } = useI18n();
  const zh = language === "zh";
  const [selectedId, setSelectedId] = useState(results[0]?.id || "");
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<CasePage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const activeId = results.some((result) => result.id === selectedId) ? selectedId : results[0]?.id || "";
  const activeResult = results.find((result) => result.id === activeId);
  const limit = 10;

  useEffect(() => {
    const controller = new AbortController();
    setPage(null);
    setError("");
    setLoading(true);
    if (!activeId) { setLoading(false); return; }
    const query = new URLSearchParams({ offset: String(offset), limit: String(limit) });
    if (metricId) query.set("metric_id", metricId);
    void fetch(`/api/v1/results/${encodeURIComponent(activeId)}/cases?${query}`, { signal: controller.signal })
      .then(async (response) => {
        const value = await response.json();
        if (!response.ok) throw new ApiError(typeof value.detail === "string" ? value.detail : response.statusText, response.status);
        return value as CasePage;
      })
      .then((value) => { if (!controller.signal.aborted) setPage(value); })
      .catch((caught) => { if (!controller.signal.aborted) setError(uiErrorMessage(caught, language)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [activeId, metricId, offset, attempt, language]);

  return (
    <section className="case-inspector" aria-label={zh ? "逐例结果" : "Example results"}>
      <div className="case-toolbar">
        <h3>{zh ? "逐例结果" : "Example results"}</h3>
        {results.length > 1 ? <label className="compact-field">
          <span>{t("run")}</span>
          <select value={activeId} onChange={(event) => { setSelectedId(event.target.value); setOffset(0); }}>
            {results.map((result) => <option key={result.id} value={result.id}>{displayTechnicalText(result.name)}</option>)}
          </select>
        </label> : null}
      </div>
      <p className="section-description">{zh ? "查看每个样本的指标变化；已保存文本的样本还可查看输入、参考答案和模型输出。" : "Inspect metric changes for each example, with inputs, references and model outputs when saved."}</p>
      {loading ? <LoadingBlock label={zh ? "正在读取样本" : "Loading examples"} /> : error ? <div>
        <InlineError message={error} />
        <Button variant="secondary" icon={RefreshCw} onClick={() => setAttempt((value) => value + 1)}>{zh ? "重新加载" : "Retry"}</Button>
      </div> : !page?.cases.length ? <EmptyState icon={FileQuestion} title={zh ? "没有已保存的逐例结果" : "No saved example results"}
        description={zh ? "可先查看汇总指标和运行日志。" : "You can review the aggregate metrics and run logs."} /> : <>
        <div className="case-list">
          {page.cases.map((item, index) => {
            const teacherForced = item.measurement_kind?.startsWith("teacher_forced") === true;
            const metric = activeResult?.metrics.find((candidate) => candidate.id === item.metric_id);
            const predictionAgreement = (item.scoring ?? metric?.scoring) === "pre_post_token_agreement";
            const nll = item.measurement_kind === "target_token_nll" || metric?.unit === "nll";
            const hasText = [item.prompt, item.expected, item.before, item.after].some((value) => typeof value === "string");
            const hasScore = [item.before_score, item.after_score, item.score].some((value) => value !== null && value !== undefined);
            const pairedScores = item.before_score !== null && item.before_score !== undefined && item.after_score !== null && item.after_score !== undefined;
            const agreementScore = item.score ?? item.after_score;
            const scoreColumns = predictionAgreement
              ? (agreementScore !== null && agreementScore !== undefined
                ? [{ label: zh ? "前后预测一致率" : "Prediction agreement", value: agreementScore, change: false }] : [])
              : pairedScores
              ? [
                { label: zh ? "更新前" : "Before", value: item.before_score!, change: false },
                { label: zh ? "更新后" : "After", value: item.after_score!, change: false },
                { label: zh ? "变化（后 − 前）" : "Change (after − before)", value: item.after_score! - item.before_score!, change: true }
              ]
              : [
                ...(item.before_score !== null && item.before_score !== undefined
                  ? [{ label: zh ? "更新前" : "Before", value: item.before_score, change: false }] : []),
                ...(item.after_score !== null && item.after_score !== undefined || item.score !== null && item.score !== undefined
                  ? [{ label: zh ? "样本指标" : "Score", value: (item.after_score ?? item.score)!, change: false }] : [])
              ];
            const scoreLabel = predictionAgreement
              ? (zh ? "局部性 · 更新前后预测一致率" : "Locality · prediction agreement before and after")
              : nll
              ? (item.group === "retain" ? (zh ? "保留集 · 目标答案 NLL" : "Retain set · target NLL") : item.group === "forget" ? (zh ? "遗忘集 · 目标答案 NLL" : "Forget set · target NLL") : (zh ? "目标答案 NLL" : "Target NLL"))
              : metric ? metricDisplayLabel(metric, activeResult?.metrics || [], language) : displayTechnicalText(item.metric_id);
            const scoreDirection = predictionAgreement
              ? (zh ? "1 表示前后逐词预测完全一致，0 表示没有一致词元；即使两次预测都不正确，也可能得到 1。" : "1 means all token predictions stayed the same; 0 means none matched. Two incorrect predictions can still score 1 if they are unchanged.")
              : nll && item.group === "retain"
              ? (zh ? "保留集：更新后接近更新前为宜；NLL 上升表示参考答案的概率下降。" : "Retain set: prefer little change; a higher NLL means a lower probability for the reference answer.")
              : nll && item.group === "forget"
                ? (zh ? "遗忘集：NLL 上升表示参考答案的概率下降；同时查看保留集变化。" : "Forget set: a higher NLL means a lower probability for the reference answer; also inspect changes on the retain set.")
                : metric ? ({ higher: t("higherBetter"), lower: t("lowerBetter"), closer_to_zero: t("closerZeroBetter"), context: t("contextOnly") })[metric.direction] : "";
            const formatScore = (value: number | null) => value === null || value === undefined ? (zh ? "未记录" : "Not recorded") : value.toLocaleString(zh ? "zh-CN" : "en-US", { maximumFractionDigits: 3 });
            const outputTitle = teacherForced ? (zh ? "逐词预测" : "Token predictions") : (zh ? "模型输出" : "Model output");
            const renderAnswer = (text: string | null, before: boolean) => typeof text === "string" ? <div className="answer-block">
              <h4>{before ? (zh ? "更新前" : "Before") : (zh ? "更新后" : "After")} · {outputTitle}</h4>
              <pre>{text || (zh ? "（空回答）" : "(Empty answer)")}</pre>
            </div> : null;
            return <article className="case-record" key={item.id}>
              <div className="subsection-title">
                <h4>{zh ? "样本" : "Example"} {offset + index + 1}</h4>
                <span className="muted">{predictionAgreement && agreementScore !== null && agreementScore !== undefined
                  ? (agreementScore === 1 ? (zh ? "预测未变" : "Predictions unchanged") : (zh ? "预测有变化" : "Predictions changed"))
                  : ({correct: zh ? "正确" : "Correct", incorrect: zh ? "不正确" : "Incorrect", measured: zh ? "已计算指标" : "Measured", recorded: zh ? "已记录输出" : "Recorded", unavailable: zh ? "结果不可用" : "Unavailable"} as Record<string, string>)[item.status] || ""}</span>
              </div>
              {typeof item.prompt === "string" ? <div className="case-prompt"><strong>{zh ? "输入" : "Input"}</strong><p>{item.prompt || (zh ? "（空文本）" : "(Empty text)")}</p></div> : null}
              {typeof item.expected === "string" ? <div className="case-prompt"><strong>{predictionAgreement ? (zh ? "参考答案（用于逐词评测）" : "Reference answer (for token evaluation)") : (zh ? "参考答案" : "Reference answer")}</strong><p>{item.expected || (zh ? "（空文本）" : "(Empty text)")}</p></div> : null}
              {teacherForced && (typeof item.before === "string" || typeof item.after === "string") ? <p className="section-description">{zh ? "逐词预测使用给定的参考上下文，与自由生成回答分别统计。" : "Token predictions use the provided reference context and are measured separately from free generation."}</p> : null}
              {typeof item.before === "string" || typeof item.after === "string" ? <div className="answer-grid">{renderAnswer(item.before, true)}{renderAnswer(item.after, false)}</div> : null}
              {hasScore ? <>
                <div className="table-scroll"><table className="data-table">
                  <caption>{scoreLabel}</caption>
                  <thead><tr>
                    {scoreColumns.map((column) => <th key={column.label}>{column.label}</th>)}
                  </tr></thead>
                  <tbody><tr>
                    {scoreColumns.map((column) => <td key={column.label} title={String(column.value)}>{column.change && column.value > 0 ? "+" : ""}{formatScore(column.value)}</td>)}
                  </tr></tbody>
                </table></div>
                {scoreDirection ? <p className="section-description">{scoreDirection}</p> : null}
              </> : null}
              {!hasText ? <p className="section-description">{zh ? "此记录仅保存逐例指标，未保存输入、参考答案或模型回答。" : "This record contains example metrics only; input, reference and model response text were not saved."}</p> : item.unavailable_reason ? <p className="section-description">{resultReasonText(item.unavailable_reason, language)}</p> : null}
              <details className="source-details"><summary>{zh ? "查看来源与评测设置" : "Source and evaluation settings"}</summary>
                <dl className="metadata-grid">
                  <div><dt>{zh ? "来源" : "Source"}</dt><dd className="mono">{item.source}{item.source_pointer ? ` · ${item.source_pointer}` : ""}</dd></div>
                  <div><dt>{zh ? "样本编号" : "Example ID"}</dt><dd>{item.case_index ?? "—"}{item.probe_index !== null && item.probe_index !== undefined ? ` / ${item.probe_index}` : ""}</dd></div>
                  <div><dt>{zh ? "评测协议" : "Evaluation protocol"}</dt><dd>{item.protocol_version ?? "—"}</dd></div>
                  <div><dt>{zh ? "评分方式" : "Scoring"}</dt><dd>{item.scoring ?? item.measurement_kind ?? "—"}</dd></div>
                  <div><dt>{zh ? "评测来源 / 阶段" : "Evaluator / phase"}</dt><dd>{item.evaluator ?? "—"} / {item.phase ?? "—"}</dd></div>
                  {item.unavailable_reason ? <div><dt>{zh ? "原始原因" : "Recorded reason"}</dt><dd className="mono">{item.unavailable_reason}</dd></div> : null}
                </dl>
                {item.warnings?.map((warning, warningIndex) => <p key={warningIndex}>{displayTechnicalText(warning)}</p>)}
              </details>
            </article>;
          })}
        </div>
        <div className="case-toolbar">
          <span>{offset + 1}–{offset + page.cases.length} / {page.total}</span>
          <div className="section-action">
            <Button variant="secondary" icon={ChevronLeft} disabled={offset === 0} onClick={() => setOffset((value) => Math.max(0, value - limit))}>{zh ? "上一页" : "Previous"}</Button>
            <Button variant="secondary" icon={ChevronRight} disabled={!page.has_more} onClick={() => setOffset((value) => value + limit)}>{zh ? "下一页" : "Next"}</Button>
          </div>
        </div>
      </>}
      {page?.warnings?.length ? <details className="source-details"><summary>{zh ? "读取说明" : "Reading notes"}</summary>{page.warnings.map((warning, index) => <p key={index}>{displayTechnicalText(warning)}</p>)}</details> : null}
    </section>
  );
}
