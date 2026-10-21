import { useEffect, useState } from "react";
import { displayTechnicalText } from "../i18n";

type Trace = {
  session_id: string;
  status: string;
  response?: string;
  error?: string;
  calls: { skill: string; status: string; command?: string; error?: string; output?: string }[];
};

export function AgentSession({ intent, provider, language }: {
  intent: string; provider: string; language: string;
}) {
  const zh = language === "zh";
  const [execute, setExecute] = useState(false);
  const [cuda, setCuda] = useState("0");
  const [trace, setTrace] = useState<Trace | null>(null);
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);
  const active = !!trace && ["pending", "running", "cancelling"].includes(trace.status);

  async function call(path: string, payload?: object) {
    const response = await fetch("/api/v1/agent/sessions" + path, payload ? {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
    } : undefined);
    const value = await response.json();
    if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : JSON.stringify(value.detail));
    return value;
  }

  useEffect(() => {
    if (!active || !trace) return;
    let disposed = false;
    const timer = window.setInterval(() => {
      call("/" + trace.session_id).then(value => { if (!disposed) setTrace(value); })
        .catch(err => { if (!disposed) setError(String(err)); });
    }, 1500);
    return () => { disposed = true; window.clearInterval(timer); };
  }, [active, trace?.session_id]);

  async function start() {
    setStarting(true); setError("");
    try {
      const result = await call("", { intent, provider, execute, cuda: execute ? cuda : null,
        max_steps: 12, max_cases: 2, timeout: 600 });
      setTrace({ ...result, calls: [] });
    } catch (err) { setError(String(err)); }
    finally { setStarting(false); }
  }

  return <details className="agent-plan">
    <summary>{zh ? "模型自主选择 Skills" : "Let the model choose Skills"}</summary>
    <p>{zh ? "模型根据每步结果继续选择技能。默认只预览，最多 12 次调用、2 条样本、10 分钟。" :
      "The model chooses each next Skill from actual observations. Preview by default; at most 12 calls, 2 cases and 10 minutes."}</p>
    <label><input type="checkbox" checked={execute} disabled={active}
      onChange={event => setExecute(event.target.checked)} />
      {zh ? "允许执行 GPU 任务" : "Allow GPU execution"}</label>
    {execute && <label>{zh ? "GPU 编号" : "GPU indices"}
      <input value={cuda} disabled={active} onChange={event => setCuda(event.target.value)}
        pattern="[0-9]+(,[0-9]+)*" aria-label={zh ? "GPU 编号" : "GPU indices"} /></label>}
    <div className="agent-actions">
      <button type="button" disabled={active || starting || provider === "local" || intent.trim().length < 3}
        onClick={start}>{zh ? "启动自主工作流" : "Start autonomous workflow"}</button>
      {active && <button type="button" onClick={() => {
        call("/" + trace!.session_id + "/stop", {}).then(() => setTrace(old => old && ({ ...old, status: "cancelling" })))
          .catch(err => setError(String(err)));
      }}>{zh ? "停止" : "Stop"}</button>}
    </div>
    {provider === "local" && <p>{zh ? "请选择已在服务端配置的 OpenAI 或 DeepSeek。" : "Select a server-configured OpenAI or DeepSeek provider."}</p>}
    {error && <p role="alert">{error}</p>}
    {trace && <div aria-live="polite">
      <p>{trace.status}</p>
      <ol>{trace.calls.map((item, index) => <li key={index}>
        <strong>{displayTechnicalText(item.skill)}</strong> · {item.status}
        {item.command && <pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{item.command}</pre>}
        {item.error && <p>{item.error}</p>}
        {item.output && <code>{item.output}</code>}
      </li>)}</ol>
      {(trace.response || trace.error) && <p>{trace.response || trace.error}</p>}
      <a href={"/api/v1/agent/sessions/" + trace.session_id} target="_blank" rel="noreferrer">
        {zh ? "打开完整 JSON 记录" : "Open full JSON trace"}</a>
    </div>}
  </details>;
}
