import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type { ComposePayload, Modality } from "../types";
import { displayTechnicalText, uiErrorMessage } from "../i18n";
import { useDraftField } from "../state/TaskDraft";
import { Button, InlineError } from "./ui";

export function AgentSession({ intent, provider, language, modality, configuration, previewHash, previewValid }: {
  intent: string; provider: string; language: "en" | "zh"; modality?: Modality;
  configuration: ComposePayload; previewHash?: string; previewValid: boolean;
}) {
  const zh = language === "zh";
  const [execute, setExecute] = useState(false);
  const [trace, setTrace] = useDraftField("agentTrace");
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);
  const startLock = useRef(false);
  const active = !!trace && ["pending", "running", "cancelling"].includes(trace.status);
  const configurationKey = JSON.stringify(configuration);

  async function call(path: string, payload?: object) {
    const response = await fetch("/api/v1/agent/sessions" + path, payload ? {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
    } : undefined);
    const value = await response.json();
    if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : JSON.stringify(value.detail));
    return value;
  }

  useEffect(() => { setExecute(false); }, [configurationKey]);
  useEffect(() => {
    if (!active || !trace) return;
    let disposed = false;
    let timer: number;
    const poll = async () => {
      try {
        const value = await call("/" + trace.session_id);
        if (!disposed) { setTrace(value); setError(""); }
      } catch (err) {
        if (!disposed) setError(uiErrorMessage(err, language));
      } finally {
        if (!disposed) timer = window.setTimeout(poll, 1500);
      }
    };
    void poll();
    return () => { disposed = true; window.clearTimeout(timer); };
  }, [active, trace?.session_id]);

  async function start() {
    if (startLock.current || active || (execute && !previewValid)) return;
    startLock.current = true;
    setStarting(true); setError("");
    try {
      const result = await call("", { intent, provider, modality, configuration,
        preview_hash: previewHash, execute, max_steps: 12, timeout: 600 });
      setTrace({ ...result, calls: [] });
    } catch (err) { setError(uiErrorMessage(err, language)); }
    finally { startLock.current = false; setStarting(false); }
  }

  const statusLabels: Record<string, string> = zh ? {
    pending: "等待开始", running: "执行中", cancelling: "正在停止", cancelled: "已停止",
    completed: "已完成", succeeded: "已完成", failed: "失败", stopped: "已停止"
  } : { pending: "Pending", running: "Running", cancelling: "Stopping", cancelled: "Stopped",
    completed: "Completed", succeeded: "Completed", failed: "Failed", stopped: "Stopped" };

  return <details className="agent-plan autonomous-agent">
    <summary>{zh ? "使用 Agent 执行当前任务" : "Use Agent for this task"}</summary>
    <p>{zh ? "Agent 使用当前任务的配置，按运行结果选择下一步技能。" :
      "The Agent uses this task’s configuration and chooses the next Skill from observed results."}</p>
    <label className="agent-execute-option"><input type="checkbox" checked={execute} disabled={active || !previewValid}
      onChange={event => setExecute(event.target.checked)} />
      {zh ? "执行已预览的配置" : "Execute the previewed configuration"}</label>
    {!previewValid && <p className="muted">{zh ? "预览通过后可启用执行。" : "Preview the configuration to enable execution."}</p>}
    <div className="agent-actions">
      <Button variant="secondary" disabled={active || provider === "local" || intent.trim().length < 3}
        loading={starting} onClick={start}>{execute ? (zh ? "启动 Agent" : "Start Agent") : (zh ? "Agent 预览" : "Agent preview")}</Button>
      {active && <Button variant="secondary" onClick={() => {
        call("/" + trace!.session_id + "/stop", {}).then(() => setTrace(old => old && ({ ...old, status: "cancelling" })))
          .catch(err => setError(uiErrorMessage(err, language)));
      }}>{zh ? "停止" : "Stop"}</Button>}
    </div>
    {provider === "local" && <p>{zh ? "选择已在服务端配置的 OpenAI 或 DeepSeek。" : "Select a server-configured OpenAI or DeepSeek provider."}</p>}
    {error && <InlineError message={error} />}
    {trace && <div className="agent-trace" aria-live="polite">
      <p>{statusLabels[trace.status] || trace.status}</p>
      <ol>{trace.calls.map((item, index) => <li key={index}>
        <strong>{displayTechnicalText(item.skill)}</strong> · {statusLabels[item.status] || item.status}
        {item.command && <pre>{item.command}</pre>}
        {item.error && <p>{item.error}</p>}
        {item.output && <code>{item.output}</code>}
      </li>)}</ol>
      {(trace.response || trace.error) && <p>{trace.response || trace.error}</p>}
      <div className="agent-actions">
        {trace.run_id && <Link to={"/runs?selected=" + encodeURIComponent(trace.run_id)}>{zh ? "查看运行与结果" : "View run and results"}</Link>}
        <a href={"/api/v1/agent/sessions/" + trace.session_id} target="_blank" rel="noreferrer">
          {zh ? "打开执行记录" : "Open execution trace"}</a>
      </div>
    </div>}
  </details>;
}
