import {
  ArrowRight,
  Bot,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  Code2,
  Eraser,
  FileCode2,
  FlaskConical,
  GitBranch,
  MessageSquareText,
  PencilLine,
  Play,
  RefreshCw,
  Sparkles,
  Syringe
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api";
import { useDraftField } from "../state/TaskDraft";
import { DataUpload } from "../components/DataUpload";
import { AgentSession } from "../components/AgentSession";
import { MetricsView } from "../components/MetricsView";
import {
  Button,
  CopyButton,
  InlineError,
  LoadingBlock,
  OperationBadge,
  SectionHeader,
  StatusBadge
} from "../components/ui";
import {
  displayTechnicalText,
  localizedSkillDescription,
  uiErrorMessage,
  useI18n
} from "../i18n";
import type {
  Catalog,
  ComposePayload,
  Demo,
  Modality,
  Operation,
  ResponseExample,
  Skill
} from "../types";

const operationIcons = {
  unlearning: Eraser,
  inject: Syringe,
  edit: PencilLine
};

const operationHelp = {
  en: {
    unlearning: "Reduce target knowledge while measuring retention.",
    inject: "Add knowledge with parameter-efficient adaptation.",
    edit: "Replace facts with localized model interventions."
  },
  zh: {
    unlearning: "降低目标知识，同时测量保留能力。",
    inject: "通过参数高效适配增加知识。",
    edit: "用局部模型干预替换已有事实。"
  }
};

const DEFAULT_OPERATION: Operation = "unlearning";
const STARTER_RECIPE_ID = "builtin-tofu-grad-ascent-gpt2";

const defaultIntents = {
  en: {
    unlearning:
      "Unlearn TOFU on GPT-2 with GradAscent for 1 training step and check retention.",
    inject:
      "Inject RuleTaker into GPT-2 with LoRA for 1 training step and check retention.",
    edit: "Edit 1 CounterFact fact on GPT-2 with ROME and evaluate locality."
  },
  zh: {
    unlearning: "使用 GradAscent 在 GPT-2 上对 TOFU 执行 1 个遗忘训练步，并检查保留能力。",
    inject: "使用 LoRA 在 GPT-2 上对 RuleTaker 执行 1 个注入训练步，并检查保留能力。",
    edit: "使用 ROME 在 GPT-2 上编辑 1 条 CounterFact 事实，并评估局部性。"
  }
};

const multimodalIntents = {
  en: {
    unlearning: "Unlearn image-text knowledge on Qwen2.5-VL-3B with GradAscent for 1 training step and review retention.",
    inject: "Inject image-text knowledge into Qwen2.5-VL-3B with LoRA for 1 training step and review retention probes.",
    edit: "Edit 1 image-text fact on Qwen2.5-VL-3B with LoRA and evaluate locality."
  },
  zh: {
    unlearning: "使用 GradAscent 在 Qwen2.5-VL-3B 上执行 1 个图文遗忘训练步，并检查保留能力。",
    inject: "使用 LoRA 在 Qwen2.5-VL-3B 上执行 1 个图文注入训练步，并检查保留探针。",
    edit: "使用 LoRA 在 Qwen2.5-VL-3B 上编辑 1 条图文知识，并评估局部性。"
  }
};

const longTextIntents = {
  en: {
    inject:
      "Inject this long-form knowledge into a suitable 7B model with representation fine-tuning, then evaluate target efficacy and retention.",
    edit:
      "Edit this long-form knowledge on a suitable 7B model with AnyEdit, then evaluate reliability, generalization, locality, and portability."
  },
  zh: {
    inject:
      "将这段长文本知识注入合适的 7B 模型，使用表示微调方法，并评估目标效果与知识保留。",
    edit:
      "在合适的 7B 模型上使用 AnyEdit 编辑这段长文本知识，并评估可靠性、泛化性、局部性与迁移性。"
  }
};

function parseOperation(value: string | null): Operation | null {
  return value === "unlearning" || value === "inject" || value === "edit"
    ? value
    : null;
}

function isUploadOperation(value: Operation): boolean {
  return value === "unlearning" || value === "inject" || value === "edit";
}

function isDefaultIntent(value: string): boolean {
  return [...Object.values(defaultIntents), ...Object.values(multimodalIntents)].some((byOperation) =>
    Object.values(byOperation).includes(value)
  );
}

function defaultTaskName(operation: Operation): string {
  if (operation !== DEFAULT_OPERATION) return `webui_${operation}_quick_run`;
  const timestamp = new Date().toISOString().replace(/\D/g, "").slice(0, 14);
  return `starter_tofu_gpt2_${timestamp}`;
}

function defaultRecipe(catalog: Catalog, operation: Operation, modality: Modality = "text") {
  if (operation === DEFAULT_OPERATION && modality === "text") {
    const starter = catalog.recipes.find(
      (item) => item.id === STARTER_RECIPE_ID && item.default_runnable
    );
    if (starter) return starter;
  }
  return catalog.recipes.find(
    (item) => item.operation === operation && (item.modality || "text") === modality && item.default_runnable
  );
}

const operationMetricKeys = {
  unlearning: "operationMetricsUnlearning",
  inject: "operationMetricsInject",
  edit: "operationMetricsEdit"
} as const;

export function Workbench() {
  const { language, t } = useI18n();
  const [searchParams] = useSearchParams();
  const initialOperation =
    parseOperation(searchParams.get("operation")) || DEFAULT_OPERATION;
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [skills, setSkills] = useState<Skill[]>([]);
  const [responseExamples, setResponseExamples] = useState<ResponseExample[]>([]);
  const [providers, setProviders] = useState<
    Array<{ id: string; label: string; configured: boolean }>
  >([]);
  const [demos, setDemos] = useState<Demo[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadAttempt, setLoadAttempt] = useState(0);
  const [error, setError] = useState("");
  const [initialized, setInitialized] = useDraftField("initialized");
  const [operation, setOperation] = useDraftField("operation");
  const [modality, setModality] = useDraftField("modality");
  const [intent, setIntent] = useDraftField("intent");
  const [provider, setProvider] = useDraftField("provider");
  const [plan, setPlan] = useDraftField("plan");
  const [recipeId, setRecipeId] = useDraftField("recipeId");
  const [experiment, setExperiment] = useDraftField("experiment");
  const [model, setModel] = useDraftField("model");
  const [trainer, setTrainer] = useDraftField("trainer");
  const [datasets, setDatasets] = useDraftField("datasets");
  const [evaluation, setEvaluation] = useDraftField("evaluation");
  const [taskName, setTaskName] = useDraftField("taskName");
  const [seed, setSeed] = useDraftField("seed");
  const [smoke, setSmoke] = useDraftField("smoke");
  const [baseSelections, setBaseSelections] = useDraftField("baseSelections");
  const [overrides, setOverrides] = useDraftField("overrides");
  const [overridesDraft, setOverridesDraft] = useDraftField("overridesDraft");
  const previousOverrides = useRef(overrides);
  useEffect(() => {
    if (previousOverrides.current !== overrides) {
      setOverridesDraft(null);
      previousOverrides.current = overrides;
    }
  }, [overrides, setOverridesDraft]);
  const parameterEdit = useMemo(() => {
    if (overridesDraft === null) return { value: overrides, error: "" };
    try {
      const value: unknown = JSON.parse(overridesDraft);
      if (!value || Array.isArray(value) || typeof value !== "object") throw new Error();
      return { value: value as Record<string, unknown>, error: "" };
    } catch {
      return { value: null, error: language === "zh" ? "请输入 JSON 对象，例如 {\"trainer.args.learning_rate\": 0.0001}。" : "Enter a JSON object, such as {\"trainer.args.learning_rate\": 0.0001}." };
    }
  }, [overridesDraft, overrides, language]);
  const [composeSnapshot, setComposeSnapshot] = useDraftField("composeSnapshot");
  const [currentRun, setCurrentRun] = useDraftField("currentRun");
  const [planning, setPlanning] = useState(false);
  const [composing, setComposing] = useState(false);
  const [running, setRunning] = useState(false);
  const [codeTab, setCodeTab] = useState<"command" | "resolved">("command");
  const [showUpload, setShowUpload] = useState(
    searchParams.get("upload") === "1" && isUploadOperation(initialOperation)
  );
  const [showSkills, setShowSkills] = useState(false);
  const uploadSectionRef = useRef<HTMLElement>(null);
  const plannedOperation = useRef<string | null>(null);
  const configuredScope = useRef(initialized ? `${operation}:${modality}` : null);
  const mounted = useRef(true);
  const planSequence = useRef(0);
  const composeSequence = useRef(0);
  const runLock = useRef(false);
  const runRequest = useRef({ key: "", id: "" });
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    const load = async () => {
      try {
        const [
          catalogValue,
          providerValue,
          demoValue,
          skillValue,
          responseExampleValue
        ] = await Promise.all([
          api.catalog(),
          api.providers(),
          api.demos(),
          api.skills(),
          api.responseExamples()
        ]);
        if (cancelled) return;
        setCatalog(catalogValue);
        setProviders(providerValue.providers);
        setDemos(demoValue.demos);
        setSkills(skillValue.skills);
        setResponseExamples(responseExampleValue.examples);
      } catch (caught) {
        if (!cancelled) {
          setError(uiErrorMessage(caught, language));
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [loadAttempt]);

  useEffect(() => {
    const requestedOperation = parseOperation(searchParams.get("operation"));
    if (requestedOperation) setOperation(requestedOperation);
    if (
      searchParams.get("upload") === "1" &&
      requestedOperation &&
      isUploadOperation(requestedOperation)
    ) {
      setShowUpload(true);
    }
  }, [searchParams]);

  useEffect(() => {
    if (!showUpload || !catalog || !isUploadOperation(operation)) return;
    const frame = window.requestAnimationFrame(() => {
      uploadSectionRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start"
      });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [catalog, operation, showUpload]);

  useEffect(() => {
    if (!catalog) return;
    const scope = `${operation}:${modality}`;
    if (configuredScope.current === scope) return;
    configuredScope.current = scope;
    setInitialized(true);
    // An explicit plan already supplied the configuration for this change.
    if (plannedOperation.current === `${operation}:${modality}`) {
      plannedOperation.current = null;
      return;
    }
    setOverrides({});
    setBaseSelections({});
    setPlan(null);
    const firstRecipe = defaultRecipe(catalog, operation, modality);
    setRecipeId(firstRecipe?.id || "");
    setExperiment(modality === "multimodal" && operation === "unlearning" ? catalog.experiments.find((item) => item.id === "unlearn/multimodal/smoke_qwen25vl3b_mllmu")?.id || "" : "");
    const firstModel =
      catalog.models.find(
        (item) => item.modality === modality && item.id === (modality === "text" ? "gpt2" : "Qwen2.5-VL-3B")
      ) || catalog.models.find((item) => item.modality === modality);
    setModel(firstModel?.id || "");
    setTrainer(catalog.trainers[operation].find((item) => item.modality === modality)?.id || "");
    setEvaluation(
      modality === "multimodal" ? (operation === "edit" ? "mm_edit" : "") : catalog.evaluations.includes(
        operation === "unlearning" ? "tofu" : operation
      )
        ? operation === "unlearning"
          ? "tofu"
          : operation
        : catalog.evaluations[0] || ""
    );
    const operationDatasets = catalog.datasets.filter((item) =>
      item.operations.includes(operation) && item.modality === modality
    );
    if (operation === "unlearning") {
      setDatasets({
        forget:
          operationDatasets.find((item) =>
            item.id.toLowerCase().includes("forget")
          )?.id || "",
        retain:
          operationDatasets.find((item) =>
            item.id.toLowerCase().includes("retain")
          )?.id || ""
      });
    } else {
      setDatasets({
        [operation === "inject" ? "train" : "edit"]:
          operationDatasets[0]?.id || ""
      });
    }
    setTaskName(defaultTaskName(operation));
    setComposeSnapshot(null);
    setCurrentRun(null);
  }, [operation, modality, catalog]);

  useEffect(() => {
    setIntent((current) =>
      !current || isDefaultIntent(current)
        ? (modality === "multimodal" ? multimodalIntents : defaultIntents)[language][operation]
        : current
    );
  }, [language, operation, modality]);

  const recipes = useMemo(
    () =>
      catalog?.recipes.filter(
        (item) => item.operation === operation && (item.modality || "text") === modality && item.default_runnable
      ) || [],
    [catalog, operation, modality]
  );

  const selectedSkillIds = useMemo(
    () => new Set(plan?.steps.map((step) => step.skill) || []),
    [plan]
  );

  const planExperiments = useMemo(
    () =>
      Object.entries(plan?.experiments || {}).filter(
        (entry): entry is [string, string] => Boolean(entry[1])
      ),
    [plan]
  );

  const responseExample = useMemo(
    () => modality === "text" ? responseExamples.find((item) => item.operation === operation) : undefined,
    [operation, responseExamples, modality]
  );

  const payload = useMemo<ComposePayload>(() => {
    const base: ComposePayload = {
      operation,
      modality,
      task_name: taskName,
      seed,
      smoke,
      overrides
    };
    if (recipeId) return { ...base, ...baseSelections, recipe_id: recipeId };
    if (experiment) return { ...base, ...baseSelections, experiment };
    return {
      ...base,
      model,
      trainer,
      datasets,
      evaluation: evaluation || null
    };
  }, [
    operation,
    modality,
    taskName,
    seed,
    smoke,
    recipeId,
    experiment,
    model,
    trainer,
    datasets,
    evaluation,
    overrides,
    baseSelections
  ]);

  const payloadKey = JSON.stringify(payload);
  const composeResult =
    overridesDraft === null && composeSnapshot?.key === payloadKey ? composeSnapshot.result : null;

  const latest = useRef({ payloadKey, intent, provider, language });
  latest.current = { payloadKey, intent, provider, language };
  const previousPayload = useRef(payloadKey);
  useEffect(() => {
    if (previousPayload.current !== payloadKey) {
      setComposeSnapshot(null);
      setCurrentRun(null);
      previousPayload.current = payloadKey;
    }
  }, [payloadKey]);

  const applyConfiguration = (value: ComposePayload) => {
    const nextModality = value.modality || "text";
    if (operation !== value.operation || modality !== nextModality) {
      plannedOperation.current = `${value.operation}:${nextModality}`;
    }
    setOperation(value.operation); setModality(nextModality);
    setRecipeId(value.recipe_id || ""); setExperiment(value.experiment || "");
    setModel(value.model || ""); setTrainer(value.trainer || "");
    setDatasets(value.datasets || {}); setEvaluation(value.evaluation || "");
    setTaskName(value.task_name); setSeed(value.seed); setSmoke(value.smoke);
    setOverrides(value.overrides || {});
    setBaseSelections({ model: value.model, trainer: value.trainer, datasets: value.datasets, evaluation: value.evaluation });
    setComposeSnapshot(null); setCurrentRun(null);
  };

  const generatePlan = async () => {
    const sequence = ++planSequence.current;
    const context = JSON.stringify(latest.current);
    setPlanning(true);
    setError("");
    try {
      const value = await api.plan({ intent, provider, language, modality });
      if (!mounted.current || sequence !== planSequence.current || JSON.stringify(latest.current) !== context) return;
      setPlan(value);
      if (value.configuration) applyConfiguration(value.configuration);
      else setError(Object.values(value.configuration_errors || {}).join("\n") || (language === "zh" ? "该计划还需要补全配置。" : "Complete the configuration for this plan."));
    } catch (caught) {
      if (mounted.current && JSON.stringify(latest.current) === context) setError(uiErrorMessage(caught, language));
    } finally {
      if (mounted.current && sequence === planSequence.current) setPlanning(false);
    }
  };

  const compose = async () => {
    if (overridesDraft !== null) return;
    const sequence = ++composeSequence.current;
    setComposing(true);
    setComposeSnapshot(null);
    setError("");
    try {
      const result = await api.compose(payload);
      if (!mounted.current || sequence !== composeSequence.current || latest.current.payloadKey !== payloadKey) return;
      setComposeSnapshot({ key: payloadKey, result });
      setCodeTab("command");
    } catch (caught) {
      if (mounted.current && latest.current.payloadKey === payloadKey) setError(uiErrorMessage(caught, language));
    } finally {
      if (mounted.current && sequence === composeSequence.current) setComposing(false);
    }
  };

  const run = async () => {
    if (!composeResult?.valid || composing || planning || runLock.current) return;
    runLock.current = true;
    if (runRequest.current.key !== payloadKey) {
      runRequest.current = { key: payloadKey, id: Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, "0")).join("") };
    }
    setRunning(true);
    setError("");
    try {
      const value = await api.startRun({ ...payload, request_id: runRequest.current.id, preview_hash: composeResult.preview_hash });
      if (latest.current.payloadKey === payloadKey) setCurrentRun(value);
    } catch (caught) {
      setError(uiErrorMessage(caught, language));
    } finally {
      runLock.current = false;
      if (mounted.current) setRunning(false);
    }
  };

  if (loading) return <LoadingBlock label={t("loading")} />;
  if (!catalog) {
    return (
      <div className="page workbench-page">
        {error ? <InlineError message={error} /> : null}
        <Button
          variant="secondary"
          icon={RefreshCw}
          onClick={() => setLoadAttempt((attempt) => attempt + 1)}
        >
          {t("refresh")}
        </Button>
      </div>
    );
  }

  const providerLabels: Record<string, string> = {
    local: t("plannerLocal"),
    openai: t("plannerOpenAI"),
    deepseek: t("plannerDeepSeek")
  };
  const planFieldLabels: Record<string, string> = {
    operation: t("operation"),
    model: t("model"),
    methods: t("method"),
    evaluation: t("evaluation")
  };
  const runStatusLabels: Record<string, string> = {
    queued: t("queued"),
    running: t("running"),
    stopping: t("stopping"),
    succeeded: t("succeeded"),
    failed: t("failed"),
    stopped: t("stopped"),
    interrupted: t("interrupted")
  };

  return (
    <div className="page workbench-page">
      <section className="workbench-heading">
        <div>
          <h1>{t("workbench")}</h1>
          <p>{operationHelp[language][operation]}</p>
        </div>
      </section>

      {error ? <InlineError message={error} /> : null}

      <div
        className="operation-segments"
        role="tablist"
        aria-label={t("operation")}
      >
        {catalog.operations.map((item) => {
          const Icon = operationIcons[item.id];
          return (
            <button
              role="tab"
              aria-selected={operation === item.id}
              className={"operation-segment segment-" + item.id}
              key={item.id}
              onClick={() => {
                setOperation(item.id);
              }}
            >
              <Icon size={18} />
              <span>{item.label[language]}</span>
              <small>{t(operationMetricKeys[item.id])}</small>
            </button>
          );
        })}
      </div>

      {responseExample ? (
        <details className="reference-example">
          <summary>{t("beforeAfterResponse")} · {responseExample.model} · {responseExample.method}</summary>
        <section className={"response-comparison response-" + operation}>
          <div className="response-comparison-head">
            <div>
              <h2>{t("beforeAfterResponse")}</h2>
            </div>
            <div className="response-example-meta">
              <OperationBadge operation={operation} />
              <span>{responseExample.model}</span>
              <span>{responseExample.method}</span>
              <StatusBadge level={responseExample.evidence_level} compact />
            </div>
          </div>
          <div className="response-prompt">
            <MessageSquareText size={17} />
            <div>
              <span>{t("probeQuestion")}</span>
              <strong>{responseExample.prompt}</strong>
              <small>
                {t("expectedAnswer")}: {responseExample.expected}
              </small>
            </div>
          </div>
          <div className="response-pair">
            <article>
              <span>{t("beforeUpdate")}</span>
              <p>{responseExample.before_response}</p>
            </article>
            <ArrowRight size={18} aria-hidden="true" />
            <article className="response-after">
              <span>{t("afterUpdate")}</span>
              <p>{responseExample.after_response}</p>
            </article>
          </div>
          <div className="response-comparison-foot">
            <p>{responseExample.outcome[language]}</p>
            <span
              title={displayTechnicalText(
                responseExample.source_artifacts.join("\n")
              )}
            >
              {responseExample.sources_available
                ? t("auditedArtifacts")
                : t("missingArtifacts")}
              {" · "}
              {displayTechnicalText(responseExample.dataset)}
            </span>
          </div>
        </section>
        </details>
      ) : null}

      <div className="workbench-grid">
        <div className="workbench-main">
          <section className="panel agent-panel">
            <SectionHeader
              title={t("planIntent")}
            />
            <textarea
              className="intent-input"
              value={intent}
              rows={4}
              placeholder={t("intentPlaceholder")}
              onChange={(event) => { setIntent(event.target.value); setPlan(null); }}
            />
            <div className="intent-presets" aria-label={t("intentPresets")}>
              <button
                type="button"
                onClick={() => {
                  setOperation("inject");
                  setModality("text");
                  setIntent(longTextIntents[language].inject);
                  setPlan(null);
                  setComposeSnapshot(null);
                }}
              >
                <Syringe size={14} />
                {t("longTextInjectionPreset")}
              </button>
              <button
                type="button"
                onClick={() => {
                  setOperation("edit");
                  setModality("text");
                  setIntent(longTextIntents[language].edit);
                  setPlan(null);
                  setComposeSnapshot(null);
                }}
              >
                <PencilLine size={14} />
                {t("longTextEditingPreset")}
              </button>
            </div>
            <div className="agent-actions">
              <label className="compact-field">
                <span>{t("provider")}</span>
                <select
                  value={provider}
                  onChange={(event) => setProvider(event.target.value)}
                >
                  {providers.map((item) => (
                    <option
                      value={item.id}
                      key={item.id}
                      disabled={!item.configured}
                    >
                      {providerLabels[item.id] || item.label}
                      {item.configured ? "" : " · " + t("notConfigured")}
                    </option>
                  ))}
                </select>
              </label>
              <Button
                icon={Sparkles}
                loading={planning}
                disabled={intent.trim().length < 3}
                onClick={generatePlan}
              >
                {t("generatePlan")}
              </Button>
            </div>

            <AgentSession intent={intent} provider={provider} language={language} modality={modality}
              configuration={payload} previewHash={composeResult?.valid ? composeResult.preview_hash : undefined}
              previewValid={!!composeResult?.valid} />
            <div className="planner-skills">
              <div className="planner-skills-header">
                <button
                  type="button"
                  aria-expanded={showSkills}
                  onClick={() => setShowSkills((value) => !value)}
                >
                  <Bot size={16} />
                  <strong>{t("availableSkills")}</strong>
                  <span>{skills.length}</span>
                  {plan ? (
                    <small>
                      {selectedSkillIds.size} {t("selectedForPlan")}
                    </small>
                  ) : null}
                  <ChevronDown
                    className={showSkills ? "is-open" : ""}
                    size={15}
                  />
                </button>
                <Link to="/extensions">
                  {t("viewRegistry")} <ChevronRight size={14} />
                </Link>
              </div>
              {showSkills && <p className="section-description">{language === "zh"
                ? "这些是项目共享的工作流定义；Agent 可调用的工具随执行模式而定。"
                : "These are shared workflow definitions. The tools available to the Agent depend on its execution mode."}</p>}
              {showSkills ? (
                <div className="planner-skill-list" role="list">
                  {skills.map((skill) => {
                    const selected = selectedSkillIds.has(skill.id);
                    return (
                      <span
                        role="listitem"
                        className={selected ? "is-selected" : ""}
                        key={skill.id}
                        title={localizedSkillDescription(
                          skill.id,
                          language,
                          skill.description
                        )}
                      >
                        {selected ? <Check size={11} /> : null}
                        {displayTechnicalText(skill.id)}
                      </span>
                    );
                  })}
                </div>
              ) : null}
            </div>

            {plan ? (
              <div className="agent-plan">
                <div className="plan-summary">
                  <div>
                    <Bot size={18} />
                    <strong>{plan.model_id}</strong>
                    <span>{plan.methods.join(" + ")}</span>
                  </div>
                  <div>
                    <OperationBadge operation={plan.operation} />
                    <span className="case-count">
                      {plan.configuration_limits?.max_samples != null
                        ? `${plan.configuration_limits.max_samples} ${t("cases")}`
                        : plan.configuration_limits?.max_steps != null
                          ? `${plan.configuration_limits.max_steps} ${language === "zh" ? "训练步" : "training steps"}`
                          : (language === "zh" ? "按配置执行" : "Configured scope")}
                    </span>
                  </div>
                </div>
                {planExperiments.length ? (
                  <div className="plan-experiments">
                    <span>{t("configuredPlanRecipes")}</span>
                    <div>
                      {planExperiments.map(([method, experimentId]) => (
                        <button
                          type="button"
                          aria-pressed={experiment === experimentId}
                          className={
                            experiment === experimentId ? "is-selected" : ""
                          }
                          key={method}
                          disabled={!plan.configurations?.[method]}
                          onClick={() => {
                            const configuration = plan.configurations?.[method];
                            if (configuration) applyConfiguration(configuration);
                          }}
                        >
                          <strong>{method}</strong>
                          <small>{displayTechnicalText(experimentId)}</small>
                        </button>
                      ))}
                    </div>
                  </div>
                ) : null}
                <ol className="skill-timeline">
                  {plan.steps.map((step, index) => (
                    <li key={step.skill}>
                      <span className="skill-index">
                        {String(index + 1).padStart(2, "0")}
                      </span>
                      <div>
                        <strong>{displayTechnicalText(step.skill)}</strong>
                        <small>{step.reason}</small>
                      </div>
                      <ChevronRight size={15} />
                    </li>
                  ))}
                </ol>
                <div className="plan-rationale">
                  <h3>{t("rationale")}</h3>
                  {plan.reasoning.map((item) => (
                    <div key={item.field}>
                      <span>{planFieldLabels[item.field] || item.field}</span>
                      <p>{item.reason}</p>
                    </div>
                  ))}
                </div>
                {plan.warnings.length ? (
                  <div className="plan-warnings">
                    <CircleAlert size={17} />
                    <div>
                      <strong>{t("warnings")}</strong>
                      {plan.warnings.map((warning) => (
                        <p key={warning}>{warning}</p>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>
            ) : null}
          </section>

          <section className="panel recipe-panel">
            <SectionHeader
              title={t("recipe")}
              action={
                <Button
                  variant="ghost"
                  icon={RefreshCw}
                  onClick={() => setComposeSnapshot(null)}
                >
                  {t("refresh")}
                </Button>
              }
            />
            <div className="form-grid">
              <label className="field field-wide">
                <span>{t("modality")}</span>
                <select value={modality} onChange={(event) => {setModality(event.target.value as Modality); setPlan(null);}}>
                  <option value="text">{language === "zh" ? "文本" : "Text"}</option>
                  <option value="multimodal">{language === "zh" ? "图文" : "Image + text"}</option>
                </select>
              </label>
              <label className="field field-wide">
                <span>{t("recipe")}</span>
                <select
                  value={experiment ? "__agent__" : recipeId || "__manual__"}
                  onChange={(event) => {
                    const value = event.target.value;
                    setExperiment(value.startsWith("experiment:") ? value.slice(11) : "");
                    setRecipeId(value.startsWith("experiment:") || value === "__manual__" || value === "__agent__" ? "" : value);
                    setOverrides({});
                    setBaseSelections({});
                    if (value === "__manual__" && recipeId) {
                      const selected = catalog.recipes.find(item => item.id === recipeId);
                      if (selected) {
                        setModel(selected.model); setTrainer(selected.trainer);
                        setDatasets(selected.datasets); setEvaluation(selected.evaluation || "");
                      }
                    }
                    setComposeSnapshot(null);
                  }}
                >
                  {experiment ? (
                    <option value="__agent__">
                      {t("agentRecipe")} · {displayTechnicalText(experiment)}
                    </option>
                  ) : null}
                  {recipes.map((recipe) => (
                    <option key={recipe.id} value={recipe.id}>
                      {recipe.model} · {recipe.trainer?.split("/").at(-1)}
                    </option>
                  ))}
                  {modality === "multimodal" ? catalog.experiments.filter((item) => item.operation === operation && item.modality === modality).map((item) => (
                    <option key={item.id} value={`experiment:${item.id}`}>{displayTechnicalText(item.id)}</option>
                  )) : null}
                  <option value="__manual__">{t("manualComposition")}</option>
                </select>
              </label>
              {!recipeId && !experiment ? (
                <>
                  <label className="field">
                    <span>{t("model")}</span>
                    <select
                      value={model}
                      onChange={(event) => setModel(event.target.value)}
                    >
                      {catalog.models
                        .filter((item) => item.modality === modality)
                        .map((item) => (
                          <option value={item.id} key={item.id}>
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
                        <option value={item.id} key={item.id}>
                          {item.id.split("/").at(-1)}
                        </option>
                      ))}
                    </select>
                  </label>
                  {Object.entries(datasets).map(([role, value]) => (
                    <label className="field" key={role}>
                      <span>
                        {t("dataset")} · {role}
                      </span>
                      <select
                        value={value}
                        onChange={(event) =>
                          setDatasets((current) => ({
                            ...current,
                            [role]: event.target.value
                          }))
                        }
                      >
                        {catalog.datasets
                          .filter((item) =>
                            item.operations.includes(operation) && item.modality === modality &&
                            (operation !== "unlearning" || !item.unlearning_role || item.unlearning_role === role)
                          )
                          .map((item) => (
                            <option value={item.id} key={item.id}>
                              {displayTechnicalText(item.id)}
                            </option>
                          ))}
                      </select>
                    </label>
                  ))}
                  <label className="field">
                    <span>{t("evaluation")}</span>
                    <select
                      value={evaluation}
                      onChange={(event) => setEvaluation(event.target.value)}
                    >
                      {modality === "multimodal" && operation !== "edit" ? <option value="">{language === "zh" ? "使用该多模态路线的探针" : "Use multimodal route probes"}</option> : null}
                      {catalog.evaluations.filter((item) => {
                        const support = catalog.evaluation_support?.find(entry => entry.id === item);
                        return support ? support.modality === modality && support.operations.includes(operation)
                          : modality === "text" ? !item.startsWith("mm_") : operation === "edit" && item.startsWith("mm_edit");
                      }).map((item) => (
                        <option value={item} key={item}>
                          {item}
                        </option>
                      ))}
                    </select>
                  </label>
                  <div className="plan-warnings field-wide">
                    <CircleAlert size={16} aria-hidden="true" />
                    <p>{t("manualCompatibilityHint")}</p>
                  </div>
                </>
              ) : null}
              <label className="field field-wide">
                <span>{t("taskName")}</span>
                <input
                  value={taskName}
                  onChange={(event) => setTaskName(event.target.value)}
                />
              </label>
              <label className="field">
                <span>{t("seed")}</span>
                <input
                  type="number"
                  min={0}
                  value={seed}
                  onChange={(event) => setSeed(Number(event.target.value))}
                />
              </label>
              <label className="toggle-field">
                <input
                  type="checkbox"
                  checked={smoke}
                  onChange={(event) => setSmoke(event.target.checked)}
                />
                <span className="toggle-track" aria-hidden="true" />
                <span>{t(operation === "edit" ? "limitEdits" : "limitSteps")}</span>
              </label>
            </div>

            <details className="parameter-editor">
              <summary>{language === "zh" ? "参数与本地路径" : "Parameters and local paths"}</summary>
              <label className="field field-wide">
                <span>{language === "zh" ? "参数覆盖（JSON）" : "Parameter overrides (JSON)"}</span>
                <textarea rows={7} spellCheck={false}
                  aria-invalid={!!parameterEdit.error}
                  aria-describedby="parameter-editor-help"
                  value={overridesDraft ?? JSON.stringify(overrides, null, 2)}
                  onChange={event => setOverridesDraft(event.target.value)} />
              </label>
              <p id="parameter-editor-help" className="section-description">{language === "zh"
                ? "可调整学习率、方法参数及模型／数据路径。字段使用 Hydra 参数名；应用后重新预览。操作、模型类型和输出目录由任务配置管理。"
                : "Adjust learning rate, method settings, and model/data paths using Hydra keys. Apply changes, then preview again. Task fields manage the operation, model type, and output directory."}</p>
              {parameterEdit.error && <InlineError message={parameterEdit.error} />}
              <Button variant="secondary" disabled={overridesDraft === null || !parameterEdit.value}
                onClick={() => { if (parameterEdit.value) { setOverrides(parameterEdit.value); setOverridesDraft(null); setComposeSnapshot(null); } }}>
                {language === "zh" ? "应用参数" : "Apply parameters"}
              </Button>
              {overridesDraft !== null && <Button variant="ghost" onClick={() => setOverridesDraft(null)}>
                {language === "zh" ? "恢复已应用参数" : "Restore applied parameters"}
              </Button>}
            </details>

            <div className="execute-bar">
              <div className="execute-actions">
                <Button
                  variant="secondary"
                  icon={GitBranch}
                  loading={composing}
                  disabled={overridesDraft !== null}
                  onClick={compose}
                >
                  {t("preview")}
                </Button>
                <Button
                  icon={Play}
                  loading={running}
                  disabled={!composeResult?.valid || composing || planning || !!currentRun}
                  onClick={run}
                >
                  {t("runNow")}
                </Button>
              </div>
            </div>

            {composeResult && !composeResult.valid && (
              <InlineError message={composeResult.stderr || (language === "zh" ? "配置未通过校验，请调整参数后重新预览。" : "Configuration validation failed. Update the parameters and preview again.")} />
            )}
            {composeResult ? (
              <div
                className={
                  "compose-output " +
                  (composeResult.valid ? "compose-valid" : "compose-invalid")
                }
              >
                <div className="code-tabs">
                  <div role="tablist">
                    <button
                      type="button"
                      role="tab"
                      aria-selected={codeTab === "command"}
                      className={codeTab === "command" ? "active" : ""}
                      onClick={() => setCodeTab("command")}
                    >
                      <Code2 size={15} /> {t("command")}
                    </button>
                    <button
                      type="button"
                      role="tab"
                      aria-selected={codeTab === "resolved"}
                      className={codeTab === "resolved" ? "active" : ""}
                      onClick={() => setCodeTab("resolved")}
                    >
                      <FileCode2 size={15} /> {t("resolved")}
                    </button>
                  </div>
                  <div className="compose-status">
                    {composeResult.valid ? (
                      <>
                        <CheckCircle2 size={15} />
                        {t("valid")}
                      </>
                    ) : (
                      <>
                        <CircleAlert size={15} />
                        {t("invalid")}
                      </>
                    )}
                    <CopyButton
                      text={
                        codeTab === "command"
                          ? composeResult.command
                          : composeResult.resolved_config
                      }
                    />
                  </div>
                </div>
                <pre>
                  <code>
                    {codeTab === "command"
                      ? composeResult.command
                      : composeResult.resolved_config || composeResult.stderr}
                  </code>
                </pre>
              </div>
            ) : null}

            {currentRun ? (
              <div className="run-started">
                <FlaskConical size={18} />
                <div>
                  <strong>{displayTechnicalText(currentRun.task_name)}</strong>
                  <span>
                    {runStatusLabels[currentRun.status] || currentRun.status} ·{" "}
                    {currentRun.id}
                  </span>
                </div>
                <Link to={"/runs?selected=" + currentRun.id}>
                  {t("runs")} <ChevronRight size={15} />
                </Link>
              </div>
            ) : null}
          </section>

          {isUploadOperation(operation) && (modality === "text" || operation === "inject") ? (
            <section
              className="panel custom-data-panel"
              ref={uploadSectionRef}
            >
              <SectionHeader
                title={
                  operation === "unlearning"
                    ? t("customUnlearningData")
                    : operation === "inject"
                      ? t("customInjectionData")
                      : t("customEditingData")
                }
                action={
                  <Button
                    variant="secondary"
                    icon={FileCode2}
                    onClick={() => setShowUpload((value) => !value)}
                  >
                    {showUpload ? t("hide") : t("uploadData")}
                  </Button>
                }
              />
              {showUpload ? (
                <DataUpload
                  key={`${operation}:${modality}`}
                  catalog={catalog}
                  seed={seed}
                  operation={operation}
                  modality={modality}
                  recommendedModel={
                    plan?.operation === operation ? plan.model_id : undefined
                  }
                  recommendedMethod={
                    plan?.operation === operation ? plan.methods[0] : undefined
                  }
                  onRegistered={(value) => {
                    applyConfiguration(value);
                    setPlan(null);
                  }}
                />
              ) : (
                <p className="section-description">
                  {operation === "inject"
                    ? t("customInjectionDataHint")
                    : operation === "edit"
                      ? t("customEditingDataHint")
                      : t("customUnlearningDataHint")}
                </p>
              )}
            </section>
          ) : null}
        </div>

        <aside className="workbench-rail">
          <section className="rail-section task-summary" aria-live="polite">
            <h2>{language === "zh" ? "当前任务" : "Current task"}</h2>
            <OperationBadge operation={operation} />
            <dl>
              <div><dt>{t("modality")}</dt><dd>{modality === "text" ? (language === "zh" ? "文本" : "Text") : (language === "zh" ? "图文" : "Image + text")}</dd></div>
              <div><dt>{t("recipe")}</dt><dd>{displayTechnicalText(experiment || recipes.find(item => item.id === recipeId)?.experiment || (language === "zh" ? "自定义配置" : "Custom configuration"))}</dd></div>
              {(!experiment || baseSelections.model) && <>
                <div><dt>{t("model")}</dt><dd>{baseSelections.model || recipes.find(item => item.id === recipeId)?.model || model}</dd></div>
                <div><dt>{t("method")}</dt><dd>{baseSelections.trainer?.split("/").at(-1) || recipes.find(item => item.id === recipeId)?.trainer?.split("/").at(-1) || trainer.split("/").at(-1)}</dd></div>
              </>}
              {Object.keys(baseSelections.datasets || {}).length > 0 && <div><dt>{t("dataset")}</dt><dd>{Object.values(baseSelections.datasets || {}).map(displayTechnicalText).join(", ")}</dd></div>}
              <div><dt>{t("seed")}</dt><dd>{seed}</dd></div>
              <div><dt>{language === "zh" ? "执行范围" : "Run scope"}</dt><dd>{smoke ? t(operation === "edit" ? "limitEdits" : "limitSteps") : (language === "zh" ? `样本 ${overrides.max_edits ?? overrides["++max_train_samples"] ?? overrides["data.max_samples"] ?? "按配置"}；步数 ${overrides["++trainer.args.max_steps"] ?? overrides["trainer.args.max_steps"] ?? "按配置"}` : `Cases: ${overrides.max_edits ?? overrides["++max_train_samples"] ?? overrides["data.max_samples"] ?? "configured"}; steps: ${overrides["++trainer.args.max_steps"] ?? overrides["trainer.args.max_steps"] ?? "configured"}`)}</dd></div>
            </dl>
            <ol className="task-progress">
              <li className="is-complete">{language === "zh" ? "设置任务" : "Configure task"}</li>
              <li className={composeResult?.valid ? "is-complete" : ""}>{t("preview")}</li>
              <li className={currentRun ? "is-complete" : ""}>{t("runNow")}</li>
              <li>{language === "zh" ? "查看效果与副作用" : "Review effects and side effects"}</li>
            </ol>
            {currentRun && <Link to={"/runs?selected=" + currentRun.id}>{t("runs")} <ChevronRight size={14} /></Link>}
          </section>

          {demos.length > 0 && <section className="rail-section demo-section">
            <div className="rail-title">
              <FlaskConical size={17} />
              <h2>{t("realDemos")}</h2>
            </div>
            <div className="demo-list">
              {demos.map((demo) => (
                <article className="demo-card" key={demo.slug}>
                  <div className="demo-card-head">
                    <OperationBadge operation={demo.result.operation} />
                    <StatusBadge
                      level={demo.result.evidence_level}
                      compact
                    />
                  </div>
                  <strong>{demo.title[language]}</strong>
                  <p>{demo.summary[language]}</p>
                  <MetricsView metrics={demo.result.metrics} compact />
                </article>
              ))}
            </div>
          </section>}
        </aside>
      </div>
    </div>
  );
}
