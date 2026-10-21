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
  Gauge,
  GitBranch,
  MessageSquareText,
  PencilLine,
  Play,
  RefreshCw,
  Sparkles,
  Syringe
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useOutletContext, useSearchParams } from "react-router-dom";
import { api } from "../api";
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
  ComposeResult,
  Demo,
  Health,
  Operation,
  Plan,
  ResponseExample,
  RunRecord,
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
      "Unlearn 1 TOFU record on GPT-2 with GradAscent and check retention.",
    inject:
      "Inject 1 RuleTaker rule into GPT-2 with LoRA and check retention.",
    edit: "Edit 1 CounterFact fact on GPT-2 with ROME and evaluate locality."
  },
  zh: {
    unlearning: "使用 GradAscent 在 GPT-2 上遗忘 1 条 TOFU 记录，并检查保留能力。",
    inject: "使用 LoRA 向 GPT-2 注入 1 条 RuleTaker 规则，并检查保留能力。",
    edit: "使用 ROME 在 GPT-2 上编辑 1 条 CounterFact 事实，并评估局部性。"
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
  return Object.values(defaultIntents).some((byOperation) =>
    Object.values(byOperation).includes(value)
  );
}

function defaultTaskName(operation: Operation): string {
  if (operation !== DEFAULT_OPERATION) return `webui_${operation}_quick_run`;
  const timestamp = new Date().toISOString().replace(/\D/g, "").slice(0, 14);
  return `starter_tofu_gpt2_${timestamp}`;
}

function defaultRecipe(catalog: Catalog, operation: Operation) {
  if (operation === DEFAULT_OPERATION) {
    const starter = catalog.recipes.find(
      (item) => item.id === STARTER_RECIPE_ID && item.default_runnable
    );
    if (starter) return starter;
  }
  return catalog.recipes.find(
    (item) => item.operation === operation && item.default_runnable
  );
}

function preferredGpu(health: Health | null): number | null {
  if (!health?.gpus.length) return null;
  const available = health.gpus
    .filter((item) => item.available)
    .sort((left, right) => right.memory_free_mib - left.memory_free_mib);
  return available[0]?.index ?? health.gpus[0].index;
}

const operationMetricKeys = {
  unlearning: "operationMetricsUnlearning",
  inject: "operationMetricsInject",
  edit: "operationMetricsEdit"
} as const;

export function Workbench() {
  const { language, t } = useI18n();
  const { health } = useOutletContext<{ health: Health | null }>();
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
  const [operation, setOperation] = useState<Operation>(initialOperation);
  const [intent, setIntent] = useState(
    defaultIntents[language][initialOperation]
  );
  const [provider, setProvider] = useState("local");
  const [plan, setPlan] = useState<Plan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [recipeId, setRecipeId] = useState("");
  const [experiment, setExperiment] = useState("");
  const [model, setModel] = useState("");
  const [trainer, setTrainer] = useState("");
  const [datasets, setDatasets] = useState<Record<string, string>>({});
  const [evaluation, setEvaluation] = useState("");
  const [taskName, setTaskName] = useState(() =>
    defaultTaskName(initialOperation)
  );
  const [seed, setSeed] = useState(42);
  const [smoke, setSmoke] = useState(true);
  const [gpu, setGpu] = useState<number | null>(null);
  const [composeSnapshot, setComposeSnapshot] = useState<{
    key: string;
    result: ComposeResult;
  } | null>(null);
  const [composing, setComposing] = useState(false);
  const [running, setRunning] = useState(false);
  const [currentRun, setCurrentRun] = useState<RunRecord | null>(null);
  const [codeTab, setCodeTab] = useState<"command" | "resolved">("command");
  const [showUpload, setShowUpload] = useState(
    searchParams.get("upload") === "1" && isUploadOperation(initialOperation)
  );
  const [showSkills, setShowSkills] = useState(true);
  const uploadSectionRef = useRef<HTMLElement>(null);
  const plannedOperation = useRef<Operation | null>(null);

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
        const firstRecipe = defaultRecipe(catalogValue, initialOperation);
        setRecipeId(firstRecipe?.id || "");
        setGpu(preferredGpu(health));
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
    // An explicit plan already supplied the recipe for this operation change.
    if (plannedOperation.current === operation) {
      plannedOperation.current = null;
      return;
    }
    const firstRecipe = defaultRecipe(catalog, operation);
    setRecipeId(firstRecipe?.id || "");
    setExperiment("");
    const firstModel =
      catalog.models.find(
        (item) => item.modality === "text" && item.id === "gpt2"
      ) || catalog.models.find((item) => item.modality === "text");
    setModel(firstModel?.id || "");
    setTrainer(catalog.trainers[operation][0]?.id || "");
    setEvaluation(
      catalog.evaluations.includes(
        operation === "unlearning" ? "tofu" : operation
      )
        ? operation === "unlearning"
          ? "tofu"
          : operation
        : catalog.evaluations[0] || ""
    );
    const operationDatasets = catalog.datasets.filter((item) =>
      item.operations.includes(operation)
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
  }, [operation, catalog]);

  useEffect(() => {
    if (gpu !== null || !health?.gpus.length) return;
    setGpu(preferredGpu(health));
  }, [health, gpu]);

  useEffect(() => {
    setIntent((current) =>
      isDefaultIntent(current)
        ? defaultIntents[language][operation]
        : current
    );
  }, [language, operation]);

  const recipes = useMemo(
    () =>
      catalog?.recipes.filter(
        (item) => item.operation === operation && item.default_runnable
      ) || [],
    [catalog, operation]
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
    () => responseExamples.find((item) => item.operation === operation),
    [operation, responseExamples]
  );

  const payload = useMemo<ComposePayload>(() => {
    const base: ComposePayload = {
      operation,
      task_name: taskName,
      seed,
      smoke,
      overrides: {}
    };
    if (recipeId) return { ...base, recipe_id: recipeId };
    if (experiment) return { ...base, experiment };
    return {
      ...base,
      model,
      trainer,
      datasets,
      evaluation
    };
  }, [
    operation,
    taskName,
    seed,
    smoke,
    recipeId,
    experiment,
    model,
    trainer,
    datasets,
    evaluation
  ]);

  const payloadKey = JSON.stringify(payload);
  const composeResult =
    composeSnapshot?.key === payloadKey ? composeSnapshot.result : null;

  useEffect(() => {
    setComposeSnapshot(null);
    setCurrentRun(null);
  }, [payloadKey]);

  const generatePlan = async () => {
    setPlanning(true);
    setError("");
    try {
      const value = await api.plan({ intent, provider, language });
      setPlan(value);
      const exactExperiment = Object.values(value.experiments).find(
        (item): item is string => Boolean(item)
      );
      if (exactExperiment && value.operation !== operation) {
        plannedOperation.current = value.operation;
      }
      setOperation(value.operation);
      if (exactExperiment) {
        setRecipeId("");
        setExperiment(exactExperiment);
      }
      setTaskName(
        "agent_" +
          value.operation +
          "_" +
          value.methods.join("_").toLowerCase() +
          "_quick_run"
      );
      setComposeSnapshot(null);
    } catch (caught) {
      setError(uiErrorMessage(caught, language));
    } finally {
      setPlanning(false);
    }
  };

  const compose = async () => {
    setComposing(true);
    setComposeSnapshot(null);
    setError("");
    try {
      const result = await api.compose(payload);
      setComposeSnapshot({ key: payloadKey, result });
      setCodeTab("command");
    } catch (caught) {
      setError(uiErrorMessage(caught, language));
    } finally {
      setComposing(false);
    }
  };

  const run = async () => {
    if (!composeResult?.valid || composing || planning) return;
    setRunning(true);
    setError("");
    try {
      const value = await api.startRun({ ...payload, gpu });
      setCurrentRun(value);
    } catch (caught) {
      setError(uiErrorMessage(caught, language));
    } finally {
      setRunning(false);
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
          <div className="eyebrow">{t("controlSurface")}</div>
          <h1>{t("workbench")}</h1>
          <p>{operationHelp[language][operation]}</p>
        </div>
        <div className="heading-stats">
          <div>
            <strong>{catalog.models.length}</strong>
            <span>{t("model")}</span>
          </div>
          <div>
            <strong>
              {Object.values(catalog.trainers).reduce(
                (sum, items) => sum + items.length,
                0
              )}
            </strong>
            <span>{t("method")}</span>
          </div>
          <div>
            <strong>{catalog.recipes.filter((item) => item.default_runnable).length}</strong>
            <span>{t("recipe")}</span>
          </div>
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
        <section className={"response-comparison response-" + operation}>
          <div className="response-comparison-head">
            <div>
              <div className="eyebrow">{t("responseComparison")}</div>
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
      ) : null}

      <div className="workbench-grid">
        <div className="workbench-main">
          <section className="panel agent-panel">
            <SectionHeader
              eyebrow={t("agentSection")}
              title={t("planIntent")}
              action={<StatusBadge level="configured" compact />}
            />
            <textarea
              className="intent-input"
              value={intent}
              rows={4}
              placeholder={t("intentPlaceholder")}
              onChange={(event) => setIntent(event.target.value)}
            />
            <div className="intent-presets" aria-label={t("intentPresets")}>
              <button
                type="button"
                onClick={() => {
                  setOperation("inject");
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

            <AgentSession intent={intent} provider={provider} language={language} />
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
                      {plan.num_cases} {t("cases")}
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
                          onClick={() => {
                            if (plan.operation !== operation) {
                              plannedOperation.current = plan.operation;
                              setOperation(plan.operation);
                            }
                            setRecipeId("");
                            setExperiment(experimentId);
                            setTaskName(
                              `agent_${plan.operation}_${method.toLowerCase()}_quick_run`
                            );
                            setComposeSnapshot(null);
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
              eyebrow={t("recipeSection")}
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
                <span>{t("recipe")}</span>
                <select
                  value={experiment ? "__agent__" : recipeId || "__manual__"}
                  onChange={(event) => {
                    const value = event.target.value;
                    setExperiment("");
                    setRecipeId(
                      value === "__manual__" || value === "__agent__"
                        ? ""
                        : value
                    );
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
                        .filter((item) => item.modality === "text")
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
                      {catalog.trainers[operation].map((item) => (
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
                            item.operations.includes(operation)
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
                      {catalog.evaluations.map((item) => (
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

            <div className="execute-bar">
              <label className="compact-field gpu-select">
                <span>GPU</span>
                <select
                  value={gpu ?? ""}
                  onChange={(event) =>
                    setGpu(
                      event.target.value === ""
                        ? null
                        : Number(event.target.value)
                    )
                  }
                >
                  <option value="">{t("auto")}</option>
                  {health?.gpus.map((item) => (
                    <option key={item.index} value={item.index}>
                      GPU {item.index} · {(item.memory_free_mib / 1024).toFixed(1)} GiB {t("free")}
                    </option>
                  ))}
                </select>
              </label>
              <div className="execute-actions">
                <Button
                  variant="secondary"
                  icon={GitBranch}
                  loading={composing}
                  onClick={compose}
                >
                  {t("preview")}
                </Button>
                <Button
                  icon={Play}
                  loading={running}
                  disabled={!composeResult?.valid || composing || planning}
                  onClick={run}
                >
                  {t("runNow")}
                </Button>
              </div>
            </div>

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

          {isUploadOperation(operation) ? (
            <section
              className="panel custom-data-panel"
              ref={uploadSectionRef}
            >
              <SectionHeader
                eyebrow={t("dataSection")}
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
                  key={operation}
                  catalog={catalog}
                  operation={operation}
                  recommendedModel={
                    plan?.operation === operation ? plan.model_id : undefined
                  }
                  recommendedMethod={
                    plan?.operation === operation ? plan.methods[0] : undefined
                  }
                  onRegistered={(value) => {
                    setRecipeId("");
                    setExperiment(value.experiment || "");
                    setTaskName(value.task_name);
                    setSeed(value.seed);
                    setSmoke(value.smoke);
                    setComposeSnapshot(null);
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
          <section className="rail-section">
            <div className="rail-title">
              <Gauge size={17} />
              <h2>{t("gpuStatus")}</h2>
            </div>
            <div className="gpu-list">
              {health?.gpus.slice(0, 10).map((item) => {
                const used =
                  (item.memory_used_mib / item.memory_total_mib) * 100;
                return (
                  <button
                    type="button"
                    aria-pressed={gpu === item.index}
                    className={
                      "gpu-row " + (gpu === item.index ? "selected" : "")
                    }
                    key={item.index}
                    onClick={() => setGpu(item.index)}
                  >
                    <div>
                      <strong>GPU {item.index}</strong>
                      <span>
                        {item.utilization_percent}% {t("utilization")}
                      </span>
                    </div>
                    <div className="gpu-meter">
                      <span style={{ width: used + "%" }} />
                    </div>
                    <small>
                      {(item.memory_free_mib / 1024).toFixed(1)} GiB {t("free")}
                    </small>
                  </button>
                );
              })}
            </div>
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
