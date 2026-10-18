import {
  Clock3,
  FileOutput,
  ListFilter,
  RefreshCw,
  TerminalSquare
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api";
import { MetricsView, ProcessTimeline } from "../components/MetricsView";
import { RunLog } from "../components/RunLog";
import {
  Button,
  EmptyState,
  InlineError,
  LoadingBlock,
  OperationBadge,
  SectionHeader,
  StatusBadge
} from "../components/ui";
import { displayTechnicalText, uiErrorMessage, useI18n } from "../i18n";
import type { ResultDetail, ResultSummary, RunRecord } from "../types";

export function Runs() {
  const { language, t } = useI18n();
  const [searchParams, setSearchParams] = useSearchParams();
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [results, setResults] = useState<ResultSummary[]>([]);
  const [selected, setSelected] = useState<RunRecord | null>(null);
  const [detail, setDetail] = useState<ResultDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("all");

  const loadRuns = async () => {
    try {
      const runPayload = await api.runs();
      setRuns(runPayload.runs);
      const requested = searchParams.get("selected");
      setSelected(
        (current) =>
          runPayload.runs.find((item) => item.id === requested) ||
          current ||
          runPayload.runs[0] ||
          null
      );
    } catch (caught) {
      setError(uiErrorMessage(caught, language));
    } finally {
      setLoading(false);
    }
  };

  const loadResults = async () => {
    try {
      const resultPayload = await api.results();
      setResults(resultPayload.results);
    } catch (caught) {
      setError(uiErrorMessage(caught, language));
    }
  };

  const load = () => {
    setError("");
    void loadRuns();
    void loadResults();
  };

  useEffect(() => {
    load();
    const timer = window.setInterval(() => void loadRuns(), 5000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return;
    }
    const summary = results.find(
      (item) =>
        item.path === selected.output_dir ||
        item.path.endsWith("/" + selected.task_name)
    );
    if (!summary) {
      setDetail(null);
      return;
    }
    void api
      .result(summary.id)
      .then(setDetail)
      .catch(() => setDetail(null));
  }, [selected?.id, selected?.status, results]);

  const visibleRuns = useMemo(
    () =>
      filter === "all"
        ? runs
        : runs.filter((item) => item.status === filter),
    [runs, filter]
  );

  if (loading) return <LoadingBlock label={t("loading")} />;

  const locale = language === "zh" ? "zh-CN" : "en-US";
  const statusLabels: Record<string, string> = {
    running: t("running"),
    succeeded: t("succeeded"),
    failed: t("failed"),
    stopped: t("stopped")
  };

  return (
    <div className="page runs-page">
      <SectionHeader
        eyebrow={t("runsEyebrow")}
        title={t("activeRuns")}
        description={t("runsDescription")}
        action={
          <Button variant="secondary" icon={RefreshCw} onClick={load}>
            {t("refresh")}
          </Button>
        }
      />
      {error ? <InlineError message={error} /> : null}

      <div className="runs-layout">
        <section className="runs-index">
          <div className="filter-bar">
            <ListFilter size={16} />
            {["all", "running", "succeeded", "failed", "stopped"].map(
              (value) => (
                <button
                  key={value}
                  className={filter === value ? "active" : ""}
                  onClick={() => setFilter(value)}
                >
                  {value === "all" ? t("all") : statusLabels[value]}
                </button>
              )
            )}
          </div>
          {visibleRuns.length ? (
            <div className="run-list">
              {visibleRuns.map((run) => (
                <button
                  className={
                    "run-list-row " +
                    (selected?.id === run.id ? "selected" : "")
                  }
                  key={run.id}
                  onClick={() => {
                    setSelected(run);
                    setSearchParams({ selected: run.id });
                  }}
                >
                  <div className="run-row-main">
                    <span className={"run-state state-" + run.status} />
                    <div>
                      <strong>{displayTechnicalText(run.task_name)}</strong>
                      <span className="mono">{run.id}</span>
                    </div>
                  </div>
                  <div className="run-row-meta">
                    <OperationBadge operation={run.operation} />
                    <span>
                      <Clock3 size={13} />
                      {new Date(run.created_at).toLocaleString(locale)}
                    </span>
                    <span>GPU {run.gpu ?? t("auto")}</span>
                  </div>
                </button>
              ))}
            </div>
          ) : (
            <EmptyState icon={TerminalSquare} title={t("noRuns")} />
          )}
        </section>

        <div className="run-detail">
          <RunLog
            run={selected}
            onChange={(value) => {
              setSelected(value);
              setRuns((current) =>
                current.map((item) =>
                  item.id === value.id ? value : item
                )
              );
            }}
          />
          {selected ? (
            <section className="run-metadata">
              <div className="subsection-title">
                <h3>{t("auditTrace")}</h3>
                <StatusBadge level={selected.evidence_level} compact />
              </div>
              <dl className="metadata-grid">
                <div>
                  <dt>{t("output")}</dt>
                  <dd className="mono">
                    {displayTechnicalText(selected.output_dir)}
                  </dd>
                </div>
                <div>
                  <dt>{t("exitCode")}</dt>
                  <dd>{selected.exit_code ?? t("pending")}</dd>
                </div>
                <div>
                  <dt>{t("started")}</dt>
                  <dd>
                    {selected.started_at
                      ? new Date(selected.started_at).toLocaleString(locale)
                      : t("pending")}
                  </dd>
                </div>
                <div>
                  <dt>{t("finished")}</dt>
                  <dd>
                    {selected.finished_at
                      ? new Date(selected.finished_at).toLocaleString(locale)
                      : t("pending")}
                  </dd>
                </div>
              </dl>
            </section>
          ) : null}
          {detail ? (
            <section className="run-result">
              <div className="subsection-title">
                <h3>{t("results")}</h3>
                <a
                  className="button button-secondary"
                  href={"/api/v1/results/" + detail.id + "/report"}
                >
                  <FileOutput size={16} />
                  {t("downloadReport")}
                </a>
              </div>
              <MetricsView metrics={detail.metrics} />
              <ProcessTimeline result={detail} />
            </section>
          ) : null}
        </div>
      </div>
    </div>
  );
}
