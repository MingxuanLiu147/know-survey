import {
  ArrowDown,
  ArrowUp,
  CircleDot,
  FileQuestion,
  Gauge,
  Target
} from "lucide-react";
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
import { useI18n } from "../i18n";
import type { Metric, ResultDetail } from "../types";
import { EmptyState, StatusBadge } from "./ui";

const categoryColors: Record<string, string> = {
  efficacy: "var(--chart-teal)",
  side_effect: "var(--chart-coral)",
  efficiency: "var(--chart-gold)",
  diagnostic: "var(--chart-blue)"
};

function Direction({ metric }: { metric: Metric }) {
  const { t } = useI18n();
  const label = {
    higher: t("higherBetter"),
    lower: t("lowerBetter"),
    closer_to_zero: t("closerZeroBetter"),
    context: t("contextOnly")
  }[metric.direction];
  const Icon =
    metric.direction === "higher"
      ? ArrowUp
      : metric.direction === "lower"
        ? ArrowDown
        : metric.direction === "closer_to_zero"
          ? Target
          : CircleDot;
  return (
    <span className="metric-direction" title={label}>
      <Icon size={13} aria-hidden="true" />
      {label}
    </span>
  );
}

export function formatMetric(metric: Metric): string {
  if (metric.value === null || metric.value === undefined) return "N/A";
  const value = metric.value;
  if (metric.unit === "multiplier") return value.toFixed(2) + "×";
  if (metric.unit === "ratio") {
    return (value * 100).toFixed(1) + "%";
  }
  if (metric.unit === "seconds") {
    return value < 60
      ? value.toFixed(1) + "s"
      : (value / 60).toFixed(1) + "m";
  }
  if (metric.unit === "GiB") return value.toFixed(2) + " GiB";
  if (Math.abs(value) >= 1000 || (Math.abs(value) < 0.001 && value !== 0)) {
    return value.toExponential(2);
  }
  return value.toFixed(3);
}

export function MetricsView({
  metrics,
  compact = false
}: {
  metrics: Metric[];
  compact?: boolean;
}) {
  const { language, t } = useI18n();
  if (!metrics.length) {
    return (
      <EmptyState
        icon={FileQuestion}
        title={t("noMetrics")}
      />
    );
  }
  const chartData = metrics
    .filter(
      (metric) =>
        metric.value !== null &&
        Number.isFinite(metric.value) &&
        metric.category !== "diagnostic" &&
        metric.unit === "ratio" && metric.value >= 0 && metric.value <= 1
    )
    .slice(0, 10)
    .map((metric) => ({
      id: metric.id,
      name: metric.label[language],
      value: metric.value! * 100,
      category: metric.category,
      formatted: formatMetric(metric)
    }));
  return (
    <div className={compact ? "metrics-view compact" : "metrics-view"}>
      <div className="metric-grid">
        {metrics.slice(0, compact ? 4 : 12).map((metric) => (
          <article className="metric-card" key={metric.id} title={[metric.evaluator, metric.phase, metric.protocol_version, metric.measurement_kind, metric.unavailable_reason].filter(Boolean).join(" · ")}>
            <div className="metric-card-top">
              <span>{metric.label[language]}</span>
              <Gauge size={15} aria-hidden="true" />
            </div>
            <strong className={metric.value === null ? "is-na" : ""}>
              {formatMetric(metric)}
            </strong>
            <div className="metric-card-foot">
              <Direction metric={metric} />
              <span>{t(metric.category)}</span>
            </div>
          </article>
        ))}
      </div>
      {!compact && chartData.length > 1 ? (
        <figure className="metric-chart" aria-label={t("metricOverview")}>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 4, right: 30, bottom: 4, left: 10 }}
            >
              <CartesianGrid
                stroke="var(--border-subtle)"
                horizontal={false}
              />
              <XAxis
                type="number"
                domain={[0, 100]}
                tickFormatter={(value: number) => `${value}%`}
                tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                dataKey="name"
                type="category"
                width={120}
                tick={{ fill: "var(--text-secondary)", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                cursor={{ fill: "var(--surface-hover)" }}
                contentStyle={{
                  background: "var(--surface-raised)",
                  border: "1px solid var(--border-strong)",
                  borderRadius: 6,
                  color: "var(--text-primary)"
                }}
                formatter={(_value, _name, item) => [
                  item.payload.formatted,
                  item.payload.name
                ]}
              />
              <Bar dataKey="value" radius={[0, 3, 3, 0]} maxBarSize={18}>
                {chartData.map((item) => (
                  <Cell
                    key={item.id}
                    fill={categoryColors[item.category]}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </figure>
      ) : null}
    </div>
  );
}

export function ProcessTimeline({ result }: { result: ResultDetail }) {
  const { language, t } = useI18n();
  const statusLabels = {
    completed: t("completed"),
    missing: t("missing"),
    failed: t("failed"),
    unknown: t("unknown")
  };
  return (
    <section className="process-section">
      <div className="subsection-title">
        <h3>{t("process")}</h3>
        <StatusBadge level={result.evidence_level} compact />
      </div>
      <ol className="process-timeline">
        {result.timeline.map((step, index) => (
          <li className={"process-step step-" + step.status} key={step.id}>
            <div className="process-index">
              {step.status === "completed" ? "✓" : index + 1}
            </div>
            <div>
              <strong>{step.label[language]}</strong>
              <span>{statusLabels[step.status]}</span>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
