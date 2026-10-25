import {
  Clock3,
  FileOutput,
  ListFilter,
  RefreshCw,
  TerminalSquare
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api";
import { MetricsView, ProcessTimeline } from "../components/MetricsView";
import { RunLog } from "../components/RunLog";
import { CaseInspector } from "../components/CaseInspector";
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
import type { ResultDetail, RunRecord } from "../types";

export function Runs() {
  const { language, t } = useI18n();
  const [searchParams, setSearchParams] = useSearchParams();
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [selected, setSelected] = useState<RunRecord | null>(null);
  const [detail, setDetail] = useState<ResultDetail | null>(null);
  const [resultState, setResultState] = useState<"loading" | "pending" | "unavailable" | "ready" | "error">("loading");
  const [resultError, setResultError] = useState("");
  const [resultAttempt, setResultAttempt] = useState(0);
  const selectedIdRef = useRef<string | undefined>(undefined);
  selectedIdRef.current = selected?.id;
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("all");
  const requestedRun = useRef(searchParams.get("selected"));
  requestedRun.current = searchParams.get("selected");
  const loadVersion = useRef(0);
  const languageRef = useRef(language);
  languageRef.current = language;

  const loadRuns = async () => {
    const version = ++loadVersion.current;
    try {
      const runPayload = await api.runs();
      if (version !== loadVersion.current) return;
      setRuns(runPayload.runs);
      setError("");
      const requested = requestedRun.current;
      setSelected(
        (current) =>
          runPayload.runs.find((item) => item.id === requested) ||
          runPayload.runs.find((item) => item.id === current?.id) ||
          runPayload.runs[0] ||
          null
      );
    } catch (caught) {
      if (version === loadVersion.current) setError(uiErrorMessage(caught, languageRef.current));
    } finally {
      if (version === loadVersion.current) setLoading(false);
    }
  };

  const load = () => {
    setError("");
    void loadRuns();
    setResultAttempt((value) => value + 1);
  };

  useEffect(() => {
    void loadRuns();
    const timer = window.setInterval(() => { void loadRuns(); }, 5000);
    return () => { loadVersion.current += 1; window.clearInterval(timer); };
  }, []);

  useEffect(() => {
    setDetail(null);
    setResultError("");
    if (!selected?.result_id) {
      setResultState("unavailable");
      return;
    }
    let cancelled = false;
    let timer: number | undefined;
    const resultId = selected.result_id;
    const terminal = ["succeeded", "failed", "stopped", "interrupted"].includes(selected.status);
    setResultState("loading");
    const pollResult = async () => {
      let again = !terminal;
      let delay = 2500;
      try {
        const value = await api.result(resultId);
        if (cancelled) return;
        setResultError("");
        if ("kind" in value && value.kind === "pending") {
          setDetail(null);
          setResultState("pending");
          again = true;
          delay = Math.max(1000, Math.min(10000, (value.retry_after_seconds || 2) * 1000));
        } else if ("kind" in value && value.kind === "unavailable") {
          setDetail(null);
          setResultState("unavailable");
          again = false;
        } else {
          setDetail(value as ResultDetail);
          setResultState("ready");
        }
      } catch (caught) {
        if (cancelled) return;
        setResultState("error");
        setResultError(uiErrorMessage(caught, language));
      } finally {
        if (!cancelled && again) timer = window.setTimeout(pollResult, delay);
      }
    };
    void pollResult();
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [selected?.id, selected?.status, selected?.result_id, resultAttempt, language]);

  const visibleRuns = useMemo(
    () =>
      filter === "all"
        ? runs
        : runs.filter((item) => item.status === filter),
    [runs, filter]
  );
  const activeDetail = detail?.id === selected?.result_id ? detail : null;

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
                    </div>
                  </div>
                  <div className="run-row-meta">
                    <OperationBadge operation={run.operation} />
                    <span>
                      <Clock3 size={13} />
                      {new Date(run.created_at).toLocaleString(locale)}
                    </span>
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
              if (selectedIdRef.current !== value.id) return;
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
          {selected && !activeDetail ? <section className="run-result" aria-live="polite">
            <h3>{t("results")}</h3>
            {resultState === "loading" ? <LoadingBlock label={language === "zh" ? "正在读取结果" : "Loading results"} />
              : resultState === "pending" ? <p className="section-description">{language === "zh" ? "任务已启动，结果生成后会自动显示。" : "The task has started. Results will appear when available."}</p>
              : resultState === "error" ? <InlineError message={resultError} />
              : <p className="section-description">{language === "zh" ? "此运行没有可读取的结果，请查看日志了解运行情况。" : "No readable results are available for this run. Check its logs for details."}</p>}
            {["error", "unavailable"].includes(resultState) ? <Button variant="secondary" icon={RefreshCw} onClick={() => setResultAttempt((value) => value + 1)}>{language === "zh" ? "重新读取结果" : "Retry results"}</Button> : null}
          </section> : null}
          {activeDetail ? (
            <section className="run-result">
              <div className="subsection-title">
                <h3>{t("results")}</h3>
                <a
                  className="button button-secondary"
                  href={"/api/v1/results/" + activeDetail.id + "/report"}
                >
                  <FileOutput size={16} />
                  {t("downloadReport")}
                </a>
              </div>
              <MetricsView metrics={activeDetail.metrics} />
              <ProcessTimeline result={activeDetail} />
              <CaseInspector key={activeDetail.id} results={[activeDetail]} />
            </section>
          ) : null}
        </div>
      </div>
    </div>
  );
}
