import { PauseCircle, TerminalSquare } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { displayTechnicalText, uiErrorMessage, useI18n } from "../i18n";
import type { RunEvent, RunRecord } from "../types";
import {
  Button,
  EmptyState,
  InlineError,
  OperationBadge,
  StatusBadge
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
  const viewport = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setEvents([]);
    setError("");
    if (!run) return;
    let cancelled = false;
    let cursor = -1;
    const poll = async () => {
      try {
        const [eventPayload, record] = await Promise.all([
          api.runEvents(run.id, cursor),
          api.run(run.id)
        ]);
        if (cancelled) return;
        if (eventPayload.events.length) {
          cursor = eventPayload.events.at(-1)?.index ?? cursor;
          setEvents((current) => [...current, ...eventPayload.events]);
        }
        onChange?.(record);
        if (!finalStates.has(record.status)) {
          window.setTimeout(poll, 850);
        }
      } catch (caught) {
        if (!cancelled) {
          setError(uiErrorMessage(caught, language));
        }
      }
    };
    void poll();
    return () => {
      cancelled = true;
    };
  }, [run?.id]);

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
            {run.id} · GPU {run.gpu ?? t("auto")}
          </span>
        </div>
        <div className="run-console-actions">
          <StatusBadge level={run.evidence_level} compact />
          {canStop ? (
            <Button
              variant="danger"
              icon={PauseCircle}
              loading={stopping}
              onClick={async () => {
                setStopping(true);
                try {
                  const value = await api.stopRun(run.id);
                  onChange?.(value);
                } catch (caught) {
                  setError(uiErrorMessage(caught, language));
                } finally {
                  setStopping(false);
                }
              }}
            >
              {t("stop")}
            </Button>
          ) : null}
        </div>
      </div>
      {error ? <InlineError message={error} /> : null}
      <div className="terminal" ref={viewport} role="log" aria-live="polite">
        <div className="terminal-command">
          <span>$</span> {displayTechnicalText(run.command)}
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
            <span>{t("loading")}...</span>
          </div>
        ) : null}
      </div>
    </section>
  );
}
