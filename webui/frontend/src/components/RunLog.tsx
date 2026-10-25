import { PauseCircle, RefreshCw, TerminalSquare } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { displayTechnicalText, uiErrorMessage, useI18n } from "../i18n";
import type { RunEvent, RunRecord } from "../types";
import {
  Button,
  EmptyState,
  InlineError,
  OperationBadge
} from "./ui";

const finalStates = new Set([
  "succeeded",
  "failed",
  "stopped",
  "interrupted"
]);

export function RunLog({
  run,
  onChange
}: {
  run: RunRecord | null;
  onChange?: (run: RunRecord) => void;
}) {
  const { language, t } = useI18n();
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [error, setError] = useState("");
  const [stopping, setStopping] = useState(false);
  const [retry, setRetry] = useState(0);
  const viewport = useRef<HTMLDivElement>(null);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const languageRef = useRef(language);
  languageRef.current = language;
  const runIdRef = useRef(run?.id);
  runIdRef.current = run?.id;

  useEffect(() => {
    setEvents([]);
    setError("");
    setStopping(false);
    if (!run) return;
    let cancelled = false;
    let cursor = -1;
    let timer: number | undefined;
    const poll = async () => {
      let continuePolling = true;
      let delay = 850;
      try {
        const [eventPayload, record] = await Promise.all([
          api.runEvents(run.id, cursor),
          api.run(run.id)
        ]);
        if (cancelled || runIdRef.current !== run.id) return;
        setError("");
        if (eventPayload.events.length) {
          const fresh = eventPayload.events.filter((event) => event.index > cursor).sort((a, b) => a.index - b.index);
          cursor = fresh.at(-1)?.index ?? cursor;
          setEvents((current) => {
            const byIndex = new Map(current.map((event) => [event.index, event]));
            for (const event of fresh) byIndex.set(event.index, event);
            return [...byIndex.values()].sort((a, b) => a.index - b.index);
          });
        }
        onChangeRef.current?.(record);
        continuePolling = !finalStates.has(record.status) || cursor < record.event_count - 1;
      } catch (caught) {
        if (!cancelled && runIdRef.current === run.id) {
          setError(uiErrorMessage(caught, languageRef.current));
        }
        delay = 2500;
      } finally {
        if (!cancelled && runIdRef.current === run.id && continuePolling) timer = window.setTimeout(poll, delay);
      }
    };
    void poll();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [run?.id, retry]);

  useEffect(() => {
    viewport.current?.scrollTo({
      top: viewport.current.scrollHeight,
      behavior: "smooth"
    });
  }, [events.length]);

  if (!run) {
    return (
      <EmptyState
        icon={TerminalSquare}
        title={t("noRuns")}
      />
    );
  }

  const canStop = ["queued", "running"].includes(run.status);
  const eventMessages: Record<string, string> = {
    "Command accepted.": t("eventCommandAccepted"),
    "Local process started.": t("eventProcessStarted"),
    "Interrupt requested.": t("eventInterruptRequested"),
    "Grace period expired; process killed.": t("eventKilled"),
    queued: t("queued"),
    running: t("running"),
    stopping: t("stopping"),
    succeeded: t("succeeded"),
    failed: t("failed"),
    stopped: t("stopped"),
    interrupted: t("interrupted")
  };
  return (
    <section className="run-console">
      <div className="run-console-head">
        <div>
          <div className="run-title-line">
            <strong>{displayTechnicalText(run.task_name)}</strong>
            <OperationBadge operation={run.operation} />
          </div>
          <span className="mono muted">
            {eventMessages[run.status] || run.status}
          </span>
        </div>
        <div className="run-console-actions">
          {canStop ? (
            <Button
              variant="danger"
              icon={PauseCircle}
              loading={stopping}
              onClick={async () => {
                setStopping(true);
                try {
                  const value = await api.stopRun(run.id);
                  if (runIdRef.current === value.id) onChangeRef.current?.(value);
                } catch (caught) {
                  if (runIdRef.current === run.id) setError(uiErrorMessage(caught, language));
                } finally {
                  if (runIdRef.current === run.id) setStopping(false);
                }
              }}
            >
              {t("stop")}
            </Button>
          ) : null}
        </div>
      </div>
      {error ? <div>
        <InlineError message={error} />
        <Button variant="secondary" icon={RefreshCw} onClick={() => setRetry((value) => value + 1)}>
          {language === "zh" ? "重新连接日志" : "Reconnect logs"}
        </Button>
      </div> : null}
      <div className="terminal" ref={viewport} role="log" aria-live="polite">
        <div className="terminal-command">
          <span>$</span> {run.command}
        </div>
        {events.map((event) => (
          <div
            className={"terminal-line terminal-" + event.type}
            key={event.index}
          >
            <time>
              {new Date(event.timestamp).toLocaleTimeString([], {
                hour12: false
              })}
            </time>
            <span>
              {displayTechnicalText(
                eventMessages[event.message || event.status || ""] ||
                  event.message ||
                  event.status ||
                  ""
              )}
            </span>
          </div>
        ))}
        {!events.length ? (
          <div className="terminal-line">
            <time>--:--:--</time>
            <span>{finalStates.has(run.status)
              ? (language === "zh" ? "此运行没有日志内容。" : "No log entries for this run.")
              : (language === "zh" ? "等待运行日志…" : "Waiting for run logs…")}</span>
          </div>
        ) : null}
      </div>
    </section>
  );
}
