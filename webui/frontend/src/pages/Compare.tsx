import {
  BarChart3,
  CheckSquare2,
  RefreshCw,
  Scale,
  Square,
  TriangleAlert
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { api } from "../api";
import { formatMetric } from "../components/MetricsView";
import { CaseInspector } from "../components/CaseInspector";
import {
  Button,
  InlineError,
  LoadingBlock,
  OperationBadge,
  SectionHeader,
  StatusBadge
} from "../components/ui";
import { displayTechnicalText, uiErrorMessage, useI18n } from "../i18n";
import type { ResultDetail, ResultSummary } from "../types";

const chartColors = [
  "var(--chart-teal)",
  "var(--chart-coral)",
  "var(--chart-gold)",
  "var(--chart-blue)",
  "var(--chart-green)",
  "var(--chart-pink)"
];

interface Comparison {
  comparable: boolean;
  fairness: Record<
    string,
    { values: unknown[]; known: boolean; equal: boolean }
  >;
  warnings: string[];
  comparison_basis?: string;
  metric_ids: string[];
  results: ResultDetail[];
}

export function Compare() {
  const { language, t } = useI18n();
  const [results, setResults] = useState<ResultSummary[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [metricId, setMetricId] = useState("");
  const [operation, setOperation] = useState("all");
  const [loading, setLoading] = useState(true);
  const [comparing, setComparing] = useState(false);
  const [error, setError] = useState("");
  const [loadAttempt, setLoadAttempt] = useState(0);
  const requestVersion = useRef(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    void api
      .results()
      .then((payload) => {
        if (cancelled) return;
        setResults(payload.results);
        setSelectedIds((current) => current.filter((id) => payload.results.some((result) => result.id === id)));
      })
      .catch((caught) => { if (!cancelled) setError(uiErrorMessage(caught, language)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [loadAttempt]);

  useEffect(() => () => { requestVersion.current += 1; }, []);

  const selectResults = (ids: string[]) => {
    requestVersion.current += 1;
    setSelectedIds(ids);
    setComparison(null);
    setMetricId("");
    setComparing(false);
    setError("");
  };

  const visible = useMemo(
    () =>
      results.filter(
        (item) =>
          operation === "all" || item.operation === operation
      ),
    [results, operation]
  );

  const runComparison = async () => {
    if (selectedIds.length < 2) return;
    setComparing(true);
    setError("");
    const version = ++requestVersion.current;
    try {
      const value = await api.compare(selectedIds);
      if (version !== requestVersion.current) return;
      setComparison(value);
      setMetricId(value.metric_ids[0] || "");
    } catch (caught) {
      if (version === requestVersion.current) setError(uiErrorMessage(caught, language));
    } finally {
      if (version === requestVersion.current) setComparing(false);
    }
  };

  const chartData = useMemo(() => {
    if (!comparison || !metricId) return [];
    return comparison.results
      .map((result, index) => {
        const metric = result.metrics.find((item) => item.id === metricId);
        return {
          name: displayTechnicalText(
            result.name.length > 26
              ? result.name.slice(0, 24) + "…"
              : result.name
          ),
          value: metric?.value,
          formatted: metric ? formatMetric(metric) : "N/A",
          color: chartColors[index % chartColors.length]
        };
      })
      .filter((item) => item.value !== null && item.value !== undefined);
  }, [comparison, metricId]);
  const chartMetrics = comparison?.results.flatMap((result) => result.metrics.filter((metric) => metric.id === metricId)) || [];
  const chartCompatible = ["unit", "scoring", "measurement_kind", "protocol_version"].every((key) =>
    new Set(chartMetrics.map((metric) => metric[key as "unit" | "scoring" | "measurement_kind" | "protocol_version"] ?? null)).size <= 1
  );

  if (loading) {
    return (
      <div className="page compare-page">
        <SectionHeader
          eyebrow={t("compareEyebrow")}
          title={t("compare")}
          description={t("compareDescription")}
          action={
            <Button icon={Scale} disabled>
              {t("compareSelected")} · 0
            </Button>
          }
        />
        <LoadingBlock label={t("loading")} />
      </div>
    );
  }

  const fairnessLabels: Record<string, string> = {
    operation: t("fairnessOperation"),
    model: t("fairnessModel"),
    dataset: t("fairnessDataset"),
    scope: t("fairnessScope"),
    evidence_level: language === "zh" ? "结果状态" : "Result status",
    model_revision: language === "zh" ? "模型版本" : "Model revision",
    request_count: language === "zh" ? "样本数" : "Sample count",
    protocol: language === "zh" ? "评测协议" : "Evaluation protocol",
    evaluation: language === "zh" ? "评测设置" : "Evaluation settings",
    budget: language === "zh" ? "运行设置" : "Run settings",
    seed: language === "zh" ? "随机种子" : "Seed",
    environment: language === "zh" ? "运行环境" : "Environment",
    complete_benchmark: language === "zh" ? "评测完成情况" : "Evaluation completion"
  };
  const conditionValue = (value: unknown): string => {
    if (value === null || value === undefined || value === "") return language === "zh" ? "未记录" : "Not recorded";
    if (typeof value === "boolean") return language === "zh" ? (value ? "是" : "否") : (value ? "Yes" : "No");
    return displayTechnicalText(typeof value === "string" ? value : JSON.stringify(value));
  };
  const comparisonNote = (text: string): string => {
    if (language !== "zh") return displayTechnicalText(text);
    if (text === "Compares recorded inputs, model revision, record counts, evaluation settings, run limits and environment.") return "按已记录的输入、模型版本、样本数、评测设置、运行设置和环境比较。";
    if (text === "Check that both runs have completed evaluation, share quality metrics, and use matching comparison settings.") return "请检查各运行是否完成评测、具有相同效果指标，并采用一致的比较条件。";
    return displayTechnicalText(text);
  };

  return (
    <div className="page compare-page">
      <SectionHeader
        eyebrow={t("compareEyebrow")}
        title={t("compare")}
        description={t("compareDescription")}
        action={
          <Button
            icon={Scale}
            loading={comparing}
            disabled={selectedIds.length < 2}
            onClick={runComparison}
          >
            {t("compareSelected")} · {selectedIds.length}
          </Button>
        }
      />
      {error ? <div><InlineError message={error} />
        <Button variant="secondary" icon={RefreshCw} onClick={() => {
          selectResults(selectedIds);
          setLoadAttempt((value) => value + 1);
        }}>{language === "zh" ? "重新加载结果" : "Reload results"}</Button>
      </div> : null}

      <div className="compare-layout">
        <section className="result-picker">
          <div className="picker-toolbar">
            <strong>{t("selectResults")}</strong>
            <div className="segmented-filter">
              {["all", "unlearning", "inject", "edit", "evaluation"].map(
                (value) => (
                  <button
                    key={value}
                    className={operation === value ? "active" : ""}
                    onClick={() => setOperation(value)}
                  >
                    {value === "all" ? t("all") : value}
                  </button>
                )
              )}
            </div>
          </div>
          <div className="result-picker-list">
            {!visible.length ? <p className="section-description">{language === "zh" ? "没有符合条件的结果。完成任务后可在这里选择对比。" : "No matching results. Complete a task to compare its results here."}</p> : null}
            {visible.map((result) => {
              const selected = selectedIds.includes(result.id);
              return (
                <button
                  type="button"
                  aria-pressed={selected}
                  className={"result-pick " + (selected ? "selected" : "")}
                  key={result.id}
                  onClick={() => selectResults(selected
                    ? selectedIds.filter((item) => item !== result.id)
                    : selectedIds.length >= 12 ? selectedIds : [...selectedIds, result.id])}
                >
                  {selected ? (
                    <CheckSquare2 size={17} />
                  ) : (
                    <Square size={17} />
                  )}
                  <div>
                    <strong>{displayTechnicalText(result.name)}</strong>
                    <span className="mono">
                      {displayTechnicalText(result.path)}
                    </span>
                  </div>
                  <OperationBadge operation={result.operation} />
                  <StatusBadge level={result.evidence_level} compact />
                </button>
              );
            })}
          </div>
        </section>

        <section className="comparison-surface">
          {comparison ? (
            <>
              <div
                className={
                  "fairness-banner " +
                  (comparison.comparable ? "is-comparable" : "is-exploratory")
                }
              >
                {comparison.comparable ? (
                  <Scale size={20} />
                ) : (
                  <TriangleAlert size={20} />
                )}
                <div>
                  <strong>
                    {comparison.comparable
                      ? t("comparable")
                      : t("exploratory")}
                  </strong>
                  <span>
                    {comparison.comparable
                      ? t("fairnessMatch")
                      : t("fairnessMismatch")}
                  </span>
                </div>
              </div>

              {comparison.comparison_basis ? <p className="section-description">{comparisonNote(comparison.comparison_basis)}</p> : null}
              {comparison.warnings.length ? <div className="comparison-notes" role="note">
                {comparison.warnings.map((warning, index) => <p key={index}>{comparisonNote(warning)}</p>)}
              </div> : null}
              <div className="fairness-grid">
                {Object.entries(comparison.fairness).map(([key, item]) => (
                  <div
                    className={
                      "fairness-cell " +
                      (item.known && item.equal ? "passes" : "fails")
                    }
                    key={key}
                  >
                    <span>{fairnessLabels[key] || key.replaceAll("_", " ")}</span>
                    <strong>
                      {item.known
                        ? item.equal
                          ? t("match")
                          : t("differs")
                        : (language === "zh" ? "未记录" : "Not recorded")}
                    </strong>
                    <details className="source-details">
                      <summary>{language === "zh" ? "查看各运行" : "View runs"}</summary>
                      {item.values.map((value, index) => <p key={index}>
                        <b>{displayTechnicalText(comparison.results[index]?.name || String(index + 1))}</b>: {conditionValue(value)}
                      </p>)}
                    </details>
                  </div>
                ))}
              </div>

              {comparison.metric_ids.length ? (
                <>
                  <div className="chart-toolbar">
                    <div>
                      <BarChart3 size={17} />
                      <strong>{t("metrics")}</strong>
                    </div>
                    <label className="compact-field">
                      <span>{t("metric")}</span>
                      <select
                        value={metricId}
                        onChange={(event) => setMetricId(event.target.value)}
                      >
                        {comparison.metric_ids.map((item) => (
                          <option value={item} key={item}>
                            {comparison.results.flatMap((result) => result.metrics).find((metric) => metric.id === item)?.label[language] || item}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  {chartCompatible ? <figure className="comparison-chart">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart
                        data={chartData}
                        margin={{ top: 12, right: 12, bottom: 70, left: 8 }}
                      >
                        <CartesianGrid
                          stroke="var(--border-subtle)"
                          vertical={false}
                        />
                        <XAxis
                          dataKey="name"
                          angle={-32}
                          textAnchor="end"
                          interval={0}
                          tick={{
                            fill: "var(--text-muted)",
                            fontSize: 11
                          }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <YAxis
                          domain={chartMetrics[0]?.unit === "ratio" ? [0, 1] : undefined}
                          tickFormatter={(value: number) => chartMetrics[0]?.unit === "ratio" ? `${Math.round(value * 100)}%` : chartMetrics[0]?.unit === "multiplier" ? `${value}×` : String(value)}
                          tick={{
                            fill: "var(--text-muted)",
                            fontSize: 11
                          }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <Tooltip
                          formatter={(_value, _name, item) => [
                            item.payload.formatted,
                            metricId
                          ]}
                          contentStyle={{
                            background: "var(--surface-raised)",
                            border: "1px solid var(--border-strong)",
                            borderRadius: 6,
                            color: "var(--text-primary)"
                          }}
                        />
                        <Bar
                          dataKey="value"
                          radius={[3, 3, 0, 0]}
                          maxBarSize={54}
                        >
                          {chartData.map((item) => (
                            <Cell key={item.name} fill={item.color} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </figure> : <p className="section-description">{language === "zh" ? "指标单位或评分方式不同，请分别查看下方数值。" : "Units or scoring methods differ. Review the values separately below."}</p>}

                  <div className="table-scroll">
                    <table className="data-table comparison-table">
                      <thead>
                        <tr>
                          <th>{t("run")}</th>
                          {comparison.metric_ids.map((item) => (
                            <th key={item}>{comparison.results.flatMap((result) => result.metrics).find((metric) => metric.id === item)?.label[language] || item}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {comparison.results.map((result) => (
                          <tr key={result.id}>
                            <td>
                              <strong>{displayTechnicalText(result.name)}</strong>
                              <StatusBadge
                                level={result.evidence_level}
                                compact
                              />
                            </td>
                            {comparison.metric_ids.map((item) => {
                              const metric = result.metrics.find(
                                (value) => value.id === item
                              );
                              return (
                                <td key={item}>
                                  {metric && metric.value !== null && metric.value !== undefined ? formatMetric(metric) : (language === "zh" ? "不可用" : "Unavailable")}
                                  {metric?.unavailable_reason ? <p className="muted">{displayTechnicalText(metric.unavailable_reason)}</p> : null}
                                  {metric ? <details className="source-details"><summary>{language === "zh" ? "指标说明" : "Metric details"}</summary>
                                    <p>{language === "zh" ? "单位" : "Unit"}: {metric.unit || "—"}</p>
                                    <p>{language === "zh" ? "分子 / 分母" : "Numerator / denominator"}: {metric.numerator ?? "—"} / {metric.denominator ?? "—"}</p>
                                    <p>{language === "zh" ? "评分方式" : "Scoring"}: {metric.scoring || metric.measurement_kind || "—"}</p>
                                    <p>{language === "zh" ? "评测协议" : "Protocol"}: {metric.protocol_version || "—"}</p>
                                    <p className="mono">{metric.source}</p>
                                  </details> : null}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <div className="no-common-metrics">
                  {t("noCommonMetrics")}
                </div>
              )}
              <CaseInspector key={comparison.results.map((result) => result.id).join(":") + ":" + metricId} results={comparison.results} metricId={metricId || undefined} />
            </>
          ) : (
            <div className="comparison-empty">
              <Scale size={32} />
              <strong>{t("compareSelected")}</strong>
              <span>{t("fairnessBeforeRanking")}</span>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
