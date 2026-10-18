import {
  BarChart3,
  CheckSquare2,
  Scale,
  Square,
  TriangleAlert
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
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

  useEffect(() => {
    void api
      .results()
      .then((payload) => {
        setResults(payload.results);
        const demoLike = payload.results
          .filter(
            (item) =>
              item.name.includes("local20") ||
              item.name.includes("smoke_v2")
          )
          .slice(0, 3)
          .map((item) => item.id);
        setSelectedIds(demoLike);
      })
      .catch((caught) => setError(uiErrorMessage(caught, language)))
      .finally(() => setLoading(false));
  }, []);

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
    try {
      const value = await api.compare(selectedIds);
      setComparison(value);
      setMetricId(value.metric_ids[0] || "");
    } catch (caught) {
      setError(uiErrorMessage(caught, language));
    } finally {
      setComparing(false);
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
    evidence_level: t("fairnessEvidence")
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
      {error ? <InlineError message={error} /> : null}

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
            {visible.map((result) => {
              const selected = selectedIds.includes(result.id);
              return (
                <button
                  type="button"
                  aria-pressed={selected}
                  className={"result-pick " + (selected ? "selected" : "")}
                  key={result.id}
                  onClick={() =>
                    setSelectedIds((current) =>
                      selected
                        ? current.filter((item) => item !== result.id)
                        : current.length >= 12
                          ? current
                          : [...current, result.id]
                    )
                  }
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
                        : "N/A"}
                    </strong>
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
                            {item}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <figure className="comparison-chart">
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
                            fontSize: 10
                          }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <YAxis
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
                  </figure>

                  <div className="table-scroll">
                    <table className="data-table comparison-table">
                      <thead>
                        <tr>
                          <th>{t("run")}</th>
                          {comparison.metric_ids.map((item) => (
                            <th key={item}>{item}</th>
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
                                  {metric ? formatMetric(metric) : "N/A"}
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
