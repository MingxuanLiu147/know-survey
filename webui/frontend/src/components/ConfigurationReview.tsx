import { useState } from "react";
import type { ConfigurationReview as Review } from "../types";
import { Button, InlineError } from "./ui";

/** The parent keys this component by review_hash so a new preview resets edits. */
export function ConfigurationReview({ report, language, confirmed, onConfirm, onApply }: {
  report: Review; language: "en" | "zh"; confirmed: boolean;
  onConfirm: (checked: boolean) => void;
  onApply: (changes: Record<string, unknown>) => void;
}) {
  const zh = language === "zh";
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [error, setError] = useState("");
  const dirty = Object.keys(draft).length > 0;
  const text = (value: unknown) => typeof value === "string" ? value : JSON.stringify(value);
  const layers = report.parameters.filter(p => /\.(layers|edit_layers|layer|layer_id|v_loss_layer)$/.test(p.key));
  const others = report.parameters.filter(p => !layers.includes(p));
  const fields = (parameters: Review["parameters"]) => parameters.map(parameter => <label className="field field-wide" key={parameter.key}>
    <span>{parameter.key.replace("trainer.method_args.", "").replace("trainer.args.", "")}
      {parameter.source === "recipe_default" ? (zh ? " · 默认值，待核对" : " · default, review required") : ""}
      {!parameter.active ? (zh ? " · 当前方法或模式不使用" : " · unused by this method or mode") : ""}</span>
    <input value={draft[parameter.key] ?? text(parameter.value)} disabled={!parameter.active}
      onChange={event => { setDraft(old => ({ ...old, [parameter.key]: event.target.value })); onConfirm(false); }} />
  </label>);
  const apply = () => {
    try {
      const changes: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(draft)) {
        const original = report.parameters.find(p => p.key === key)?.value;
        changes[key] = typeof original === "string" || (original === null && value !== "null") ? value : JSON.parse(value);
      }
      setError(""); onApply(changes);
    } catch { setError(zh ? "数字、布尔值和列表需使用 JSON 格式，例如 [4,5,6]；路径填写文本。" : "Use JSON for numbers, booleans and lists, e.g. [4,5,6]; enter paths as text."); }
  };
  return <section className="parameter-editor" aria-label={zh ? "运行前参数确认" : "Parameter review"}>
    <h3>{zh ? "运行前检查并确认" : "Review before running"}</h3>
    <p>{report.model.source} · {report.method} · {zh ? "模型总层数" : "Model depth"}: {report.model.num_hidden_layers ?? (zh ? "待核实" : "unavailable")}</p>
    <p className="section-description">{zh
      ? "层号从 0 开始。默认值是起点，请按当前模型和算法核对或修改。更换模型、算法、数据或参数后需要重新预览和确认。"
      : "Layer indices start at 0. Check or change defaults for this model and method. Changes to models, methods, data or parameters require a new preview and confirmation."}</p>
    {fields(layers)}
    <details><summary>{zh ? "其他算法与训练参数" : "Other method and training parameters"}</summary>{fields(others)}</details>
    {dirty && <Button variant="secondary" onClick={apply}>{zh ? "应用修改，随后重新预览" : "Apply changes, then preview again"}</Button>}
    {error && <InlineError message={error} />}
    <dl className="metadata-grid">{report.datasets.map(dataset => <div key={dataset.key}>
      <dt>{dataset.split || (zh ? "数据" : "Data")} · {dataset.count} {dataset.count_kind === "source_records"
        ? (zh ? "条原始记录" : "source records") : (zh ? "条编辑样本" : "edit cases")}</dt>
      <dd>{dataset.path}</dd><dd className="mono">SHA256: {dataset.sha256.slice(0, 16)}…</dd>
    </div>)}</dl>
    {report.errors.map((issue, i) => <InlineError key={i} message={issue[language]} />)}
    {report.warnings.map((issue, i) => <p key={i} role="status">{issue[language]}</p>)}
    <label className="agent-execute-option"><input type="checkbox" checked={confirmed}
      disabled={!report.valid || dirty} onChange={event => onConfirm(event.target.checked)} />
      {zh ? "我已检查并确认当前模型、层数、算法参数及数据" : "I have checked this model, layers, method settings and data"}</label>
  </section>;
}
