import {
  Blocks,
  Bot,
  Boxes,
  Database,
  Eraser,
  PencilLine,
  Search,
  ServerCog,
  Syringe,
  Workflow
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import {
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
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    void Promise.all([api.catalog(), api.skills()])
      .then(([catalogValue, skillValue]) => {
        setCatalog(catalogValue);
        setSkills(skillValue.skills);
      })
      .catch((caught) => setError(uiErrorMessage(caught, language)))
      .finally(() => setLoading(false));
  }, []);

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

  if (loading || !catalog) {
    return <LoadingBlock label={t("loading")} />;
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
        <div className="catalog-tabs" role="tablist">
          {(["skills", "models", "datasets", "methods"] as Tab[]).map(
            (value) => {
              const Icon = tabIcons[value];
              return (
                <button
                  role="tab"
                  aria-selected={tab === value}
                  className={tab === value ? "active" : ""}
                  key={value}
                  onClick={() => setTab(value)}
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
                    <strong>{displayTechnicalText(skill.id)}</strong>
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
                          <strong>{displayTechnicalText(item.id)}</strong>
                    </td>
                    <td>
                      {item.modality === "text"
                        ? t("textModality")
                        : t("multimodalModality")}
                    </td>
                    <td className="mono truncate-cell">
                      {item.source || t("processorPathContract")}
                    </td>
                    <td>
                      {item.local ? t("localPath") : t("registryReference")}
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
                      <strong>{displayTechnicalText(item.id)}</strong>
                    </td>
                    <td className="mono">{item.handlers.join(", ") || "N/A"}</td>
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
                      <strong>{item.id.split("/").at(-1)}</strong>
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
                    <td className="mono">configs/trainer/{item.id}.yaml</td>
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

      {!filteredSkills.length &&
      !filteredModels.length &&
      !filteredDatasets.length &&
      !filteredMethods.length ? (
        <EmptyState icon={Boxes} title={t("noMatchingRegistry")} />
      ) : null}

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
