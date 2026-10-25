import {
  CheckCircle2,
  Database,
  FileJson,
  UploadCloud
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { displayTechnicalText, uiErrorMessage, useI18n } from "../i18n";
import type { Catalog, ComposePayload, Modality, Operation } from "../types";
import { Button, InlineError } from "./ui";

type UploadOperation = Operation;

const defaultMappings: Record<UploadOperation, string> = {
  unlearning: JSON.stringify({
    id: "id",
    prompt: "question",
    target_old: "answer",
    rephrase: "rephrase_prompts",
    locality: "locality_inputs"
  }, null, 2),
  inject: JSON.stringify({
    id: "id",
    prompt: "instruction",
    input: "input",
    target_new: "output"
  }, null, 2),
  edit: JSON.stringify({
    id: "id",
    prompt: "prompt",
    subject: "subject",
    target_old: "target_old",
    target_new: "target_new",
    rephrase: "rephrase_prompts",
    locality: "locality_inputs",
    portability: "portability_inputs"
  }, null, 2)
};

function recommendedTrainerId(
  operation: UploadOperation,
  method?: string
): string | undefined {
  if (!method) return undefined;
  if (operation === "unlearning") return method.replace("unlearn/", "");
  return method.includes("/") ? method : `${operation}/${method}`;
}

export function DataUpload({
  catalog,
  operation,
  modality = "text",
  recommendedModel,
  recommendedMethod,
  onRegistered
}: {
  catalog: Catalog;
  operation: UploadOperation;
  modality?: Modality;
  recommendedModel?: string;
  recommendedMethod?: string;
  onRegistered: (payload: ComposePayload) => void;
}) {
  const { language, t } = useI18n();
  const fileInput = useRef<HTMLInputElement>(null);
  const retainFileInput = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [retainFile, setRetainFile] = useState<File | null>(null);
  const [mapping, setMapping] = useState(modality === "multimodal" ? JSON.stringify({id: "id", prompt: "prompt", target_new: "target_new", images: "images"}, null, 2) : defaultMappings[operation]);
  const [knowledgeType, setKnowledgeType] = useState(modality === "multimodal" ? "multimodal" : "fact");
  const [previewing, setPreviewing] = useState(false);
  const [registering, setRegistering] = useState(false);
  const [error, setError] = useState("");
  const [preview, setPreview] = useState<{
    upload_id: string;
    records: number;
    preview: Array<Record<string, unknown>>;
    field_mapping: Record<string, string>;
    source_sha256: string;
  } | null>(null);
  const [datasetId, setDatasetId] = useState(
    operation === "unlearning"
      ? "my_unlearning"
      : operation === "inject"
        ? "my_injection"
        : "my_edit"
  );
  const [model, setModel] = useState(
    catalog.models.find((item) => item.id === recommendedModel && item.modality === modality)?.id ||
    catalog.models.find((item) => item.id === (modality === "text" ? "gpt2" : "Qwen2.5-VL-3B"))?.id ||
      catalog.models.find((item) => item.modality === modality)?.id ||
      ""
  );
  const [trainer, setTrainer] = useState(
    catalog.trainers[operation].find(
      (item) =>
        item.id === recommendedTrainerId(operation, recommendedMethod) && item.modality === modality
    )?.id ||
      catalog.trainers[operation].find(
        (item) =>
          item.id ===
          (operation === "unlearning"
            ? "GradAscent"
            : operation === "inject"
              ? (modality === "text" ? "inject/LoRA" : "inject/MM_LoRA")
              : "edit/FT")
      )?.id ||
      catalog.trainers[operation].find((item) => item.modality === modality)?.id ||
      ""
  );
  const [registered, setRegistered] = useState<{
    dataset_id: string;
    experiment: string;
    command: string;
  } | null>(null);

  const columns = useMemo(() => {
    if (!preview?.preview.length) return [];
    return Array.from(
      new Set(preview.preview.flatMap((row) => Object.keys(row)))
    ).slice(0, 8);
  }, [preview]);
  const knowledgeTypeLabels: Record<string, string> = {
    fact: t("fact"),
    event: t("event"),
    procedural: t("procedural"),
    rule: t("rule"),
    free_text: t("freeText"),
    concept: t("concept"),
    multimodal: language === "zh" ? "图文知识" : "Image-text knowledge"
  };

  useEffect(() => {
    if (
      recommendedModel &&
      catalog.models.some((item) => item.id === recommendedModel && item.modality === modality)
    ) {
      setModel(recommendedModel);
    }
    const trainerId = recommendedTrainerId(operation, recommendedMethod);
    if (
      trainerId &&
      catalog.trainers[operation].some((item) => item.id === trainerId && item.modality === modality)
    ) {
      setTrainer(trainerId);
    }
  }, [catalog, operation, modality, recommendedMethod, recommendedModel]);

  const chooseFile = (value: File | null, role: "primary" | "retain") => {
    if (role === "retain") setRetainFile(value);
    else setFile(value);
    setPreview(null);
    setRegistered(null);
    setError("");
  };

  const runPreview = async () => {
    if (!file || (operation === "unlearning" && !retainFile)) return;
    setPreviewing(true);
    setError("");
    try {
      JSON.parse(mapping);
      const form = new FormData();
      form.append("file", file);
      if (retainFile) form.append("retain_file", retainFile);
      form.append("mapping", mapping);
      form.append("knowledge_type", knowledgeType);
      form.append("format_type", "custom");
      form.append("modality", modality);
      setPreview(await api.previewUpload(operation, form));
    } catch (caught) {
      setError(uiErrorMessage(caught, language));
    } finally {
      setPreviewing(false);
    }
  };

  const register = async () => {
    if (!preview) return;
    setRegistering(true);
    setError("");
    try {
      const value = await api.registerUpload(operation, {
        upload_id: preview.upload_id,
        modality,
        dataset_id: datasetId,
        model,
        trainer,
        evaluation: modality === "multimodal" ? null : operation === "unlearning" ? "tofu" : operation,
        seed: 42
      });
      setRegistered(value);
      onRegistered(value.compose_request);
    } catch (caught) {
      setError(uiErrorMessage(caught, language));
    } finally {
      setRegistering(false);
    }
  };

  return (
    <section className="upload-workflow">
      <div className="upload-steps" aria-label={t("uploadWorkflow")}>
        <span
          className={
            file && (operation !== "unlearning" || retainFile)
              ? "is-complete"
              : "is-current"
          }
        >
          <b>1</b> {t("uploadData")}
        </span>
        <span className={preview ? "is-complete" : file ? "is-current" : ""}>
          <b>2</b> {t("previewData")}
        </span>
        <span className={registered ? "is-complete" : preview ? "is-current" : ""}>
          <b>3</b> {t("registerData")}
        </span>
      </div>

      <div className="upload-grid">
        <div className="upload-controls">
          <button
            className={"drop-zone " + (file ? "has-file" : "")}
            type="button"
            onClick={() => fileInput.current?.click()}
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              chooseFile(event.dataTransfer.files[0] || null, "primary");
            }}
          >
            {file ? <FileJson size={24} /> : <UploadCloud size={24} />}
            <strong>
              {file?.name ||
                (operation === "unlearning" ? t("forgetData") : t("uploadData"))}
            </strong>
            <span>
              {file
                ? (file.size / 1024).toFixed(1) + " KiB"
                : t("uploadHint")}
            </span>
          </button>
          <input
            ref={fileInput}
            className="visually-hidden"
            type="file"
            accept=".json,.jsonl,.csv,application/json,text/csv"
            onChange={(event) =>
              chooseFile(event.target.files?.[0] || null, "primary")
            }
          />

          {operation === "unlearning" ? (
            <>
              <button
                className={"drop-zone " + (retainFile ? "has-file" : "")}
                type="button"
                onClick={() => retainFileInput.current?.click()}
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();
                  chooseFile(event.dataTransfer.files[0] || null, "retain");
                }}
              >
                {retainFile ? <FileJson size={24} /> : <UploadCloud size={24} />}
                <strong>{retainFile?.name || t("retainData")}</strong>
                <span>
                  {retainFile
                    ? (retainFile.size / 1024).toFixed(1) + " KiB"
                    : t("uploadHint")}
                </span>
              </button>
              <input
                ref={retainFileInput}
                className="visually-hidden"
                type="file"
                accept=".json,.jsonl,.csv,application/json,text/csv"
                onChange={(event) =>
                  chooseFile(event.target.files?.[0] || null, "retain")
                }
              />
            </>
          ) : null}

          <label className="field">
            <span>{t("fieldMapping")}</span>
            <textarea
              className="mono mapping-input"
              value={mapping}
              rows={8}
              onChange={(event) => {
                setMapping(event.target.value);
                setPreview(null);
              }}
              spellCheck={false}
            />
            <small className="field-hint">
              {modality === "multimodal" ? (language === "zh" ? "图文 JSON/JSONL 需要 prompt、target_new 和 images。图片须已存在于服务器，使用绝对路径或相对仓库路径。" : "Image-text records need prompt, target_new, and images. Images must exist on the server; use absolute or repository-relative paths.") : operation === "inject"
                ? t("injectionRequiredFields")
                : operation === "edit"
                  ? t("editingRequiredFields")
                  : t("unlearningRequiredFields")}
            </small>
          </label>
          <label className="field">
            <span>{t("knowledgeType")}</span>
            <select
              value={knowledgeType}
              onChange={(event) => setKnowledgeType(event.target.value)}
            >
              {(modality === "multimodal" ? ["multimodal"] : ["fact", "event", "procedural", "rule", "free_text", "concept"]).map(
                (value) => (
                  <option value={value} key={value}>
                    {knowledgeTypeLabels[value]}
                  </option>
                )
              )}
            </select>
          </label>
          <Button
            icon={FileJson}
            variant="secondary"
            loading={previewing}
            disabled={!file || (operation === "unlearning" && !retainFile)}
            onClick={runPreview}
          >
            {t("previewData")}
          </Button>
        </div>

        <div className="upload-preview">
          {error ? <InlineError message={error} /> : null}
          {preview ? (
            <>
              <div className="preview-summary">
                <div>
                  <strong>
                    {preview.records} {t("records")}
                  </strong>
                  <span className="mono">
                    SHA-256 {preview.source_sha256.slice(0, 16)}…
                  </span>
                </div>
                <CheckCircle2 size={20} aria-hidden="true" />
              </div>
              <div className="table-scroll">
                <table className="data-table preview-table">
                  <thead>
                    <tr>
                      {columns.map((column) => (
                        <th key={column}>{column}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.preview.map((row, index) => (
                      <tr key={String(row.id || index)}>
                        {columns.map((column) => (
                          <td key={column}>
                            {typeof row[column] === "object"
                              ? JSON.stringify(row[column])
                              : String(row[column] ?? "")}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="register-grid">
                <label className="field">
                  <span>{t("datasetId")}</span>
                  <input
                    value={datasetId}
                    onChange={(event) => setDatasetId(event.target.value)}
                    pattern="[a-z][a-z0-9_-]*"
                  />
                </label>
                <label className="field">
                  <span>{t("model")}</span>
                  <select
                    value={model}
                    onChange={(event) => setModel(event.target.value)}
                  >
                    {catalog.models
                      .filter((item) => item.modality === modality)
                      .map((item) => (
                        <option key={item.id} value={item.id}>
                          {item.id}
                        </option>
                      ))}
                  </select>
                </label>
                <label className="field">
                  <span>{t("method")}</span>
                  <select
                    value={trainer}
                    onChange={(event) => setTrainer(event.target.value)}
                  >
                    {catalog.trainers[operation].filter((item) => item.modality === modality).map((item) => (
                      <option key={item.id} value={item.id}>
                        {operation === "unlearning"
                          ? item.id
                          : item.id.replace(operation + "/", "")}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <Button
                icon={Database}
                loading={registering}
                disabled={!datasetId || !model || !trainer}
                onClick={register}
              >
                {t("registerData")}
              </Button>
            </>
          ) : (
            <div className="upload-placeholder">
              <FileJson size={28} />
              <strong>{t("previewData")}</strong>
              <span>
                {t("allRecordsValidated")}
              </span>
            </div>
          )}
          {registered ? (
            <div className="registered-result">
              <CheckCircle2 size={18} />
              <div>
                <strong>{t("registered")}</strong>
                <span>
                  {displayTechnicalText(registered.dataset_id)} ·{" "}
                  {displayTechnicalText(registered.experiment)}
                </span>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}
