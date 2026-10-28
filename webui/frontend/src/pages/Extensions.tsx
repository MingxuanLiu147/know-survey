import {
  Blocks,
  Bot,
  Boxes,
  Database,
  Eraser,
  PencilLine,
  RefreshCw,
  Search,
  ServerCog,
  Syringe,
  Workflow
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import {
  Button,
  EmptyState,
  InlineError,
  LoadingBlock,
  SectionHeader,
  StatusBadge
} from "../components/ui";
import {
  displayTechnicalText,
  localizedSkillDescription,
  uiErrorMessage,
  useI18n
} from "../i18n";
import type { Catalog, Skill } from "../types";

type Tab = "skills" | "models" | "datasets" | "methods";
const tabs: Tab[] = ["skills", "models", "datasets", "methods"];

const tabIcons = {
  skills: Bot,
  models: ServerCog,
  datasets: Database,
  methods: Workflow
};

export function Extensions() {
  const { language, t } = useI18n();
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [skills, setSkills] = useState<Skill[]>([]);
  const [tab, setTab] = useState<Tab>("skills");
  const tabButtons = useRef<Partial<Record<Tab, HTMLButtonElement | null>>>({});
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [loadAttempt, setLoadAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    void Promise.all([api.catalog(), api.skills()])
      .then(([catalogValue, skillValue]) => {
        if (cancelled) return;
        setCatalog(catalogValue);
        setSkills(skillValue.skills);
      })
      .catch((caught) => {
        if (!cancelled) setError(uiErrorMessage(caught, language));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [loadAttempt, language]);

  const filteredSkills = useMemo(
    () =>
      skills.filter(
        (item) =>
          item.id.toLowerCase().includes(search.toLowerCase()) ||
          localizedSkillDescription(item.id, language, item.description)
            .toLowerCase()
            .includes(search.toLowerCase())
      ),
    [skills, search, language]
  );

  if (loading && !catalog) {
    return <LoadingBlock label={t("loading")} />;
  }
  if (!catalog) {
    return (
      <div className="page extensions-page">
        <SectionHeader title={t("extensions")} action={
          <Button variant="secondary" icon={RefreshCw} onClick={() => setLoadAttempt((value) => value + 1)}>
            {language === "zh" ? "重新加载" : "Retry"}
          </Button>
        } />
        <InlineError message={error || (language === "zh" ? "组件目录暂时无法加载。" : "The component catalog is unavailable.")} />
      </div>
    );
  }

  const filteredModels = catalog.models.filter((item) =>
    item.id.toLowerCase().includes(search.toLowerCase())
  );
  const filteredDatasets = catalog.datasets.filter((item) =>
    item.id.toLowerCase().includes(search.toLowerCase())
  );
  const methods = Object.entries(catalog.trainers).flatMap(
    ([operation, values]) =>
      values.map((item) => ({ ...item, operation }))
  );
  const filteredMethods = methods.filter((item) =>
    item.id.toLowerCase().includes(search.toLowerCase())
  );
  const categoryLabels: Record<string, string> = {
    foundation: t("categoryFoundation"),
    extension: t("categoryExtension"),
    multimodal: t("categoryMultimodal"),
    analysis: t("categoryAnalysis"),
    execution: t("categoryExecution"),
    data: t("categoryData"),
    ui: t("categoryUi"),
    reporting: t("categoryReporting"),
    release: t("categoryRelease"),
    reproducibility: t("categoryReproducibility"),
    validation: t("categoryValidation")
  };
  const flow = [
    t("configStep"),
    t("adapterStep"),
    t("composeStep"),
    t("smokeStep"),
    t("evidenceStep")
  ];

  return (
    <div className="page extensions-page">
      <SectionHeader
        eyebrow={t("extensionsEyebrow")}
        title={t("extensions")}
        description={t("extensionsDescription")}
        action={<Button variant="secondary" icon={RefreshCw} loading={loading} onClick={() => setLoadAttempt((value) => value + 1)}>{t("refresh")}</Button>}
      />
      {error ? <InlineError message={error} /> : null}

      <section className="catalog-summary">
        <div>
          <Bot size={20} />
          <strong>{skills.length}</strong>
          <span>{t("skills")}</span>
        </div>
        <div>
          <ServerCog size={20} />
          <strong>{catalog.models.length}</strong>
          <span>{t("model")}</span>
        </div>
        <div>
          <Database size={20} />
          <strong>{catalog.datasets.length}</strong>
          <span>{t("dataset")}</span>
        </div>
        <div>
          <Workflow size={20} />
          <strong>{methods.length}</strong>
          <span>{t("method")}</span>
        </div>
      </section>

      <div className="catalog-toolbar">
        <div className="catalog-tabs" role="tablist" aria-label={t("extensions")}>
          {tabs.map(
            (value) => {
              const Icon = tabIcons[value];
              return (
                <button
                  ref={(element) => { tabButtons.current[value] = element; }}
                  id={`registry-tab-${value}`}
                  role="tab"
                  aria-selected={tab === value}
                  aria-controls="registry-panel"
                  tabIndex={tab === value ? 0 : -1}
                  className={tab === value ? "active" : ""}
                  key={value}
                  onClick={() => setTab(value)}
                  onKeyDown={(event) => {
                    const index = tabs.indexOf(value);
                    const next = event.key === "ArrowRight" ? tabs[(index + 1) % tabs.length]
                      : event.key === "ArrowLeft" ? tabs[(index + tabs.length - 1) % tabs.length]
                      : event.key === "Home" ? tabs[0]
                      : event.key === "End" ? tabs[tabs.length - 1] : null;
                    if (!next) return;
                    event.preventDefault();
                    setTab(next);
                    tabButtons.current[next]?.focus();
                  }}
                >
                  <Icon size={16} />
                  {value === "skills" ? t("skills") : t(value.slice(0, -1) as "model" | "dataset" | "method")}
                </button>
              );
            }
          )}
        </div>
        <label className="search-field">
          <Search size={16} />
          <input
            type="search"
            placeholder={t("filterRegistry")}
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </label>
      </div>

      <div id="registry-panel" role="tabpanel" aria-labelledby={`registry-tab-${tab}`} tabIndex={0}>
      {tab === "skills" ? (
        <section className="skill-catalog">
          <div className="skill-grid">
            {filteredSkills.map((skill, index) => (
              <article className="skill-card" key={skill.id}>
                <div className="skill-card-number">
                  {String(index + 1).padStart(2, "0")}
                </div>
                <div className="skill-card-body">
                  <div>
                    <strong className="notranslate" translate="no">{displayTechnicalText(skill.id)}</strong>
                    <span>{categoryLabels[skill.category] || skill.category}</span>
                  </div>
                  <p>
                    {localizedSkillDescription(
                      skill.id,
                      language,
                      skill.description
                    )}
                  </p>
                  <div className="skill-card-foot">
                    <StatusBadge level={skill.status} compact />
                    <span>
                      {skill.has_runtime
                        ? t("executableWrapper")
                        : t("contractOnly")}
                    </span>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      {tab === "models" ? (
        <section className="registry-table-section">
          <p className="muted">
            {language === "zh"
              ? "已配置表示项目提供了配置文件；运行前仍需检查本地权重和所选方法的兼容性。此列表不检查模型是否已下载。"
              : "Configured means a configuration file is provided. Check local weights and compatibility with the selected method before running. This list does not check whether weights are downloaded."}
          </p>
          <div className="table-scroll">
            <table className="data-table registry-table">
              <thead>
                <tr>
                  <th>{t("modelId")}</th>
                  <th>{t("modality")}</th>
                  <th>{t("weightSource")}</th>
                  <th>{t("cache")}</th>
                  <th>{t("status")}</th>
                </tr>
              </thead>
              <tbody>
                {filteredModels.map((item) => (
                  <tr key={item.id}>
                    <td>
                      <strong className="notranslate" translate="no" title={item.id}>{displayTechnicalText(item.id)}</strong>
                    </td>
                    <td>
                      {item.modality === "text"
                        ? t("textModality")
                        : t("multimodalModality")}
                    </td>
                    <td>
                      {item.source ? (
                        <span className="mono notranslate" translate="no" title={item.source}>{item.source}</span>
                      ) : t("processorPathContract")}
                    </td>
                    <td>
                      {!item.source
                        ? (language === "zh" ? "未列出" : "Not listed")
                        : item.local ? t("localPath") : t("registryReference")}
                    </td>
                    <td>
                      <StatusBadge level={item.status} compact />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {tab === "datasets" ? (
        <section className="registry-table-section">
          <div className="table-scroll">
            <table className="data-table registry-table">
              <thead>
                <tr>
                  <th>{t("datasetId")}</th>
                  <th>{t("handlers")}</th>
                  <th>{t("operations")}</th>
                  <th>{t("status")}</th>
                </tr>
              </thead>
              <tbody>
                {filteredDatasets.map((item) => (
                  <tr key={item.id}>
                    <td>
                      <strong className="notranslate" translate="no">{displayTechnicalText(item.id)}</strong>
                    </td>
                    <td className="mono notranslate" translate="no">{item.handlers.join(", ") || "N/A"}</td>
                    <td>
                      <div className="operation-list">
                        {item.operations.map((operation) => (
                          <span
                            className={"operation-badge operation-" + operation}
                            key={operation}
                          >
                            {operation}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td>
                      <StatusBadge level={item.status} compact />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {tab === "methods" ? (
        <section className="registry-table-section">
          <div className="table-scroll">
            <table className="data-table registry-table">
              <thead>
                <tr>
                  <th>{t("methodId")}</th>
                  <th>{t("operation")}</th>
                  <th>{t("contract")}</th>
                  <th>{t("status")}</th>
                </tr>
              </thead>
              <tbody>
                {filteredMethods.map((item) => (
                  <tr key={item.operation + item.id}>
                    <td>
                      <strong className="notranslate" translate="no">{item.id.split("/").at(-1)}</strong>
                    </td>
                    <td>
                      <span
                        className={
                          "operation-badge operation-" + item.operation
                        }
                      >
                        {item.operation}
                      </span>
                    </td>
                    <td className="mono notranslate" translate="no">configs/trainer/{item.id}.yaml</td>
                    <td>
                      <StatusBadge level={item.status} compact />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {!({skills: filteredSkills, models: filteredModels, datasets: filteredDatasets, methods: filteredMethods}[tab]).length ? (
        <EmptyState icon={Boxes} title={t("noMatchingRegistry")} />
      ) : null}
      </div>

      <section className="extension-contract">
        <div>
          <div className="eyebrow">{t("customExtensionContract")}</div>
          <h2>{t("registryFirstIntegration")}</h2>
          <p>{t("extensionDescription")}</p>
        </div>
        <div className="contract-flow" aria-label={t("customExtensionContract")}>
          {flow.map((item, index) => (
              <span key={item}>
                <b>{index + 1}</b>
                {item}
              </span>
            ))}
        </div>
        <div className="extension-contract-actions">
          <Link
            className="button button-secondary"
            to="/?operation=unlearning&upload=1"
          >
            <Eraser size={16} />
            {t("uploadUnlearningData")}
          </Link>
          <Link
            className="button button-primary"
            to="/?operation=inject&upload=1"
          >
            <Syringe size={16} />
            {t("uploadInjectionData")}
          </Link>
          <Link
            className="button button-secondary"
            to="/?operation=edit&upload=1"
          >
            <PencilLine size={16} />
            {t("uploadEditingData")}
          </Link>
        </div>
      </section>
    </div>
  );
}
