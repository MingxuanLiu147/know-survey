import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode
} from "react";

export type Language = "en" | "zh";

const messages = {
  en: {
    workbench: "Workbench",
    runs: "Runs",
    compare: "Compare",
    extensions: "Extensions",
    ready: "System ready",
    agentFirst: "Agent-first knowledge surgery",
    evidenceAware: "Evidence-aware",
    planIntent: "Describe the knowledge change",
    intentPlaceholder:
      "Example: Edit 2 CounterFact facts on Qwen2.5-1.5B with ROME, then evaluate locality.",
    generatePlan: "Generate plan",
    planning: "Planning",
    operation: "Operation",
    recipe: "Recipe",
    model: "Model",
    method: "Method",
    dataset: "Dataset",
    evaluation: "Evaluation",
    taskName: "Task name",
    smoke: "Quick validation",
    seed: "Seed",
    preview: "Compose",
    runNow: "Run on GPU",
    composing: "Composing",
    command: "Equivalent CLI",
    resolved: "Resolved config",
    plan: "Agent plan",
    rationale: "Recommendation rationale",
    warnings: "Warnings",
    gpuStatus: "GPU status",
    free: "free",
    busy: "busy",
    customData: "Custom data",
    customUnlearningData: "Custom unlearning data",
    customInjectionData: "Custom injection data",
    customEditingData: "Custom editing data",
    uploadUnlearningData: "Upload unlearning data",
    uploadInjectionData: "Upload injection data",
    uploadEditingData: "Upload editing data",
    uploadData: "Upload JSON, JSONL, or CSV",
    fieldMapping: "Field mapping",
    injectionRequiredFields: "Required: prompt, target_new",
    editingRequiredFields: "Required: prompt, subject, target_new",
    unlearningRequiredFields: "Both files require: prompt, target_old",
    forgetData: "Upload forget data",
    retainData: "Upload retain data",
    previewData: "Preview normalized data",
    registerData: "Register dataset",
    registered: "Dataset registered",
    activeRuns: "Active and recent runs",
    noRuns: "No WebUI runs yet",
    liveLogs: "Live process log",
    stop: "Stop",
    results: "Results",
    realDemos: "Real local demo artifacts",
    metrics: "Metrics",
    process: "Process trace",
    selectResults: "Select 2-12 results",
    compareSelected: "Compare selected",
    comparable: "Comparable protocol",
    exploratory: "Exploratory only",
    skills: "Agent Skills",
    catalog: "Capability catalog",
    configured: "Configured",
    local_validation: "Local validation",
    verified_smoke: "Verified run",
    verified_result: "Verified result",
    planned: "Planned",
    noMetrics: "No machine-readable metrics",
    loading: "Loading",
    error: "Error",
    refresh: "Refresh",
    copy: "Copy",
    copied: "Copied",
    downloadReport: "Download report",
    language: "中文",
    light: "Light theme",
    dark: "Dark theme",
    provider: "Planner",
    records: "records",
    evidence: "Evidence",
    all: "All",
    efficacy: "Target efficacy",
    side_effect: "Side effects",
    efficiency: "Efficiency",
    diagnostic: "Diagnostics",
    skipToContent: "Skip to content",
    primaryNavigation: "Primary navigation",
    collapseNavigation: "Collapse navigation",
    expandNavigation: "Expand navigation",
    openNavigation: "Open navigation",
    closeNavigation: "Close navigation",
    switchLanguage: "Switch to Chinese",
    auto: "auto",
    controlSurface: "KNOW-SURGERY / CONTROL SURFACE",
    agentSection: "01 / AGENT",
    recipeSection: "02 / RECIPE",
    dataSection: "03 / DATA",
    cases: "cases",
    notConfigured: "not configured",
    manualComposition: "Manual composition",
    manualCompatibilityHint:
      "Composition validates Hydra wiring only. Model mappings, method checkpoints or statistics, local data, and optional packages must also be available.",
    agentRecipe: "Agent",
    hide: "Hide",
    valid: "Valid",
    invalid: "Invalid",
    hydraComposition: "Hydra composition",
    boundedTrace: "Bounded local trace",
    realModelArtifacts: "Real model and artifacts",
    completeProtocol: "Complete protocol",
    responseComparison: "AUDITED RESPONSE PROBE",
    beforeAfterResponse: "Before / after model response",
    probeQuestion: "Probe question",
    expectedAnswer: "Expected answer",
    beforeUpdate: "Before update",
    afterUpdate: "After update",
    auditedArtifacts: "Audited source artifacts",
    missingArtifacts: "Source artifacts unavailable",
    customDataHint:
      "Convert text knowledge to JSON, JSONL, or CSV, review it, then compose.",
    customInjectionDataHint:
      "Map prompt and target_new, review every normalized record, then register a runnable injection recipe.",
    customEditingDataHint:
      "Map prompt, subject, target_new, and optional probes, then register a runnable editing recipe.",
    customUnlearningDataHint:
      "Upload separate forget and retain files, review both roles, then register a bounded unlearning recipe.",
    intentPresets: "Intent presets",
    longTextInjectionPreset: "Long-text injection · 7B · LoReFT",
    longTextEditingPreset: "Long-text editing · 7B · AnyEdit",
    availableSkills: "Available Agent Skills",
    selectedForPlan: "selected for this plan",
    viewRegistry: "View registry",
    configuredPlanRecipes: "Configured plan recipes",
    plannerLocal: "Deterministic local",
    plannerOpenAI: "OpenAI GPT",
    plannerDeepSeek: "DeepSeek",
    operationMetricsUnlearning: "forget efficacy · retention",
    operationMetricsInject: "target efficacy · retention",
    operationMetricsEdit: "reliability · generalization",
    runsEyebrow: "EXECUTION / TRACE",
    runsDescription:
      "Every WebUI action retains the exact argv, compose result, process events, and output location.",
    auditTrace: "Audit trace",
    output: "Output",
    exitCode: "Exit code",
    started: "Started",
    finished: "Finished",
    pending: "Pending",
    queued: "Queued",
    running: "Running",
    stopping: "Stopping",
    succeeded: "Succeeded",
    failed: "Failed",
    stopped: "Stopped",
    interrupted: "Interrupted",
    completed: "Completed",
    missing: "Missing",
    unknown: "Unknown",
    eventCommandAccepted: "Command accepted.",
    eventProcessStarted: "Local process started.",
    eventInterruptRequested: "Interrupt requested.",
    eventKilled: "Grace period expired; process killed.",
    compareEyebrow: "ANALYSIS / FAIRNESS GATE",
    compareDescription:
      "Ranking is enabled only when operation, model, dataset scope, protocol, and evidence level match.",
    fairnessMatch: "Fairness fields match; leaderboard ranking is permitted.",
    fairnessMismatch:
      "Exploratory view only: one or more fairness fields are missing or differ.",
    match: "Match",
    differs: "Differs",
    metric: "Metric",
    run: "Run",
    noCommonMetrics:
      "No common machine-readable metric exists across these runs.",
    fairnessBeforeRanking:
      "The fairness gate runs before any leaderboard is shown.",
    fairnessOperation: "Operation",
    fairnessModel: "Model",
    fairnessDataset: "Dataset",
    fairnessScope: "Scope",
    fairnessEvidence: "Evidence level",
    extensionsEyebrow: "REGISTRIES / CONTRACTS",
    extensionsDescription:
      "Capabilities are loaded from Hydra configs and Agent Skill contracts. Configured does not mean benchmark-verified.",
    filterRegistry: "Filter registry",
    executableWrapper: "Executable wrapper",
    contractOnly: "Contract only",
    modelId: "Model ID",
    modality: "Modality",
    weightSource: "Weight source",
    cache: "Cache",
    status: "Status",
    datasetId: "Dataset ID",
    handlers: "Handlers",
    operations: "Operations",
    methodId: "Method ID",
    contract: "Contract",
    textModality: "Text",
    multimodalModality: "Multimodal",
    processorPathContract: "processor/model path contract",
    localPath: "local path",
    registryReference: "registry reference",
    noMatchingRegistry: "No matching registry item",
    customExtensionContract: "CUSTOM EXTENSION CONTRACT",
    registryFirstIntegration: "Registry-first integration",
    extensionDescription:
      "Add a config, adapter or trainer, public fixture, compose test, and one bounded real validation run before upgrading compatibility evidence.",
    configStep: "Config",
    adapterStep: "Adapter",
    composeStep: "Compose",
    smokeStep: "Quick validation",
    evidenceStep: "Evidence",
    categoryFoundation: "Foundation",
    categoryExtension: "Extension",
    categoryMultimodal: "Multimodal",
    categoryAnalysis: "Analysis",
    categoryExecution: "Execution",
    categoryData: "Data",
    categoryUi: "UI",
    categoryReporting: "Reporting",
    categoryRelease: "Release",
    categoryReproducibility: "Reproducibility",
    categoryValidation: "Validation",
    uploadWorkflow: "Upload workflow",
    knowledgeType: "Knowledge type",
    uploadHint: "UTF-8 · JSON / JSONL / CSV · 50 MiB max",
    allRecordsValidated:
      "All records are validated before a runnable config is created.",
    fact: "Fact",
    event: "Event",
    procedural: "Procedural",
    rule: "Rule",
    freeText: "Free text",
    concept: "Concept",
    higherBetter: "Higher is better",
    lowerBetter: "Lower is better",
    closerZeroBetter: "Closer to zero is better",
    contextOnly: "Context only",
    metricOverview: "Metric overview chart",
    utilization: "util",
    requestFailed: "Request failed. Check the configuration and server logs."
  },
  zh: {
    workbench: "工作台",
    runs: "运行",
    compare: "对比",
    extensions: "扩展",
    ready: "系统就绪",
    agentFirst: "智能体优先的知识手术",
    evidenceAware: "证据感知",
    planIntent: "描述你的知识更新目标",
    intentPlaceholder:
      "示例：使用 ROME 在 Qwen2.5-1.5B 上编辑 2 条 CounterFact 事实，并评估局部性。",
    generatePlan: "生成计划",
    planning: "规划中",
    operation: "操作",
    recipe: "运行配方",
    model: "模型",
    method: "方法",
    dataset: "数据集",
    evaluation: "评估",
    taskName: "任务名",
    smoke: "快速验证",
    seed: "随机种子",
    preview: "组合配置",
    runNow: "在 GPU 上运行",
    composing: "配置组合中",
    command: "等价 CLI",
    resolved: "解析后配置",
    plan: "智能体计划",
    rationale: "推荐理由",
    warnings: "风险提示",
    gpuStatus: "GPU 状态",
    free: "空闲",
    busy: "占用",
    customData: "自定义数据",
    customUnlearningData: "自定义遗忘数据",
    customInjectionData: "自定义注入数据",
    customEditingData: "自定义编辑数据",
    uploadUnlearningData: "上传遗忘数据",
    uploadInjectionData: "上传注入数据",
    uploadEditingData: "上传编辑数据",
    uploadData: "上传 JSON、JSONL 或 CSV",
    fieldMapping: "字段映射",
    injectionRequiredFields: "必需字段：prompt、target_new",
    editingRequiredFields: "必需字段：prompt、subject、target_new",
    unlearningRequiredFields: "两份文件均需：prompt、target_old",
    forgetData: "上传遗忘集",
    retainData: "上传保留集",
    previewData: "预览规范化数据",
    registerData: "注册数据集",
    registered: "数据集已注册",
    activeRuns: "当前与近期运行",
    noRuns: "尚无界面运行记录",
    liveLogs: "实时进程日志",
    stop: "停止",
    results: "结果",
    realDemos: "真实本地演示产物",
    metrics: "指标",
    process: "过程轨迹",
    selectResults: "选择 2-12 个结果",
    compareSelected: "对比所选结果",
    comparable: "协议可比",
    exploratory: "仅探索展示",
    skills: "智能体技能",
    catalog: "能力目录",
    configured: "已配置",
    local_validation: "本地验证",
    verified_smoke: "已验证运行",
    verified_result: "已验证结果",
    planned: "规划中",
    noMetrics: "暂无机器可读指标",
    loading: "加载中",
    error: "错误",
    refresh: "刷新",
    copy: "复制",
    copied: "已复制",
    downloadReport: "下载报告",
    language: "EN",
    light: "浅色主题",
    dark: "深色主题",
    provider: "规划器",
    records: "条记录",
    evidence: "证据等级",
    all: "全部",
    efficacy: "目标效果",
    side_effect: "副作用",
    efficiency: "效率",
    diagnostic: "诊断",
    skipToContent: "跳转到主要内容",
    primaryNavigation: "主导航",
    collapseNavigation: "收起导航",
    expandNavigation: "展开导航",
    openNavigation: "打开导航",
    closeNavigation: "关闭导航",
    switchLanguage: "切换为英文",
    auto: "自动",
    controlSurface: "KNOW-SURGERY / 控制台",
    agentSection: "01 / 智能体",
    recipeSection: "02 / 运行配方",
    dataSection: "03 / 数据",
    cases: "条用例",
    notConfigured: "未配置",
    manualComposition: "手工组合",
    manualCompatibilityHint:
      "配置组合只校验 Hydra 接线；模型层映射、方法 checkpoint 或统计量、本地数据和可选依赖也必须可用。",
    agentRecipe: "智能体",
    hide: "收起",
    valid: "有效",
    invalid: "无效",
    hydraComposition: "Hydra 配置组合",
    boundedTrace: "有界本地轨迹",
    realModelArtifacts: "真实模型与产物",
    completeProtocol: "完整协议",
    responseComparison: "可审计回答探针",
    beforeAfterResponse: "模型更新前后回答",
    probeQuestion: "探针问题",
    expectedAnswer: "期望答案",
    beforeUpdate: "更新前",
    afterUpdate: "更新后",
    auditedArtifacts: "可审计来源产物",
    missingArtifacts: "来源产物不可用",
    customDataHint:
      "请先将文本知识转换为 JSON、JSONL 或 CSV，审核后再组合配置。",
    customInjectionDataHint:
      "映射 prompt 与 target_new，审核全部规范化记录后，注册可运行的注入配方。",
    customEditingDataHint:
      "映射 prompt、subject、target_new 与可选探针后，注册可运行的编辑配方。",
    customUnlearningDataHint:
      "分别上传遗忘集与保留集，审核两种角色后，注册有界遗忘配方。",
    intentPresets: "目标模板",
    longTextInjectionPreset: "长文本注入 · 7B · LoReFT",
    longTextEditingPreset: "长文本编辑 · 7B · AnyEdit",
    availableSkills: "可用智能体技能",
    selectedForPlan: "项用于当前计划",
    viewRegistry: "查看注册表",
    configuredPlanRecipes: "已配置的计划配方",
    plannerLocal: "确定性本地规划器",
    plannerOpenAI: "OpenAI GPT",
    plannerDeepSeek: "DeepSeek",
    operationMetricsUnlearning: "遗忘效果 · 保留能力",
    operationMetricsInject: "目标效果 · 保留能力",
    operationMetricsEdit: "可靠性 · 泛化性",
    runsEyebrow: "执行 / 轨迹",
    runsDescription:
      "每次界面操作都会保留完整参数、配置组合结果、进程事件和输出位置。",
    auditTrace: "审计轨迹",
    output: "输出位置",
    exitCode: "退出码",
    started: "开始时间",
    finished: "完成时间",
    pending: "等待中",
    queued: "已排队",
    running: "运行中",
    stopping: "正在停止",
    succeeded: "已成功",
    failed: "已失败",
    stopped: "已停止",
    interrupted: "已中断",
    completed: "已完成",
    missing: "缺失",
    unknown: "未知",
    eventCommandAccepted: "命令已接受。",
    eventProcessStarted: "本地进程已启动。",
    eventInterruptRequested: "已请求中断。",
    eventKilled: "等待超时，进程已终止。",
    compareEyebrow: "分析 / 公平性门禁",
    compareDescription:
      "只有操作、模型、数据范围、评估协议和证据等级一致时才允许排序。",
    fairnessMatch: "公平性字段一致，可以进行排行榜排序。",
    fairnessMismatch: "仅供探索：一个或多个公平性字段缺失或不一致。",
    match: "一致",
    differs: "不同",
    metric: "指标",
    run: "运行",
    noCommonMetrics: "这些运行之间没有共同的机器可读指标。",
    fairnessBeforeRanking: "显示任何排行榜前都会先执行公平性检查。",
    fairnessOperation: "操作",
    fairnessModel: "模型",
    fairnessDataset: "数据集",
    fairnessScope: "数据范围",
    fairnessEvidence: "证据等级",
    extensionsEyebrow: "注册表 / 接口契约",
    extensionsDescription:
      "能力来自 Hydra 配置和智能体技能契约。已配置不等于已经完成基准验证。",
    filterRegistry: "筛选注册项",
    executableWrapper: "可执行封装",
    contractOnly: "仅接口契约",
    modelId: "模型 ID",
    modality: "模态",
    weightSource: "权重来源",
    cache: "缓存",
    status: "状态",
    datasetId: "数据集 ID",
    handlers: "处理器",
    operations: "操作",
    methodId: "方法 ID",
    contract: "接口契约",
    textModality: "文本",
    multimodalModality: "多模态",
    processorPathContract: "处理器与模型路径契约",
    localPath: "本地路径",
    registryReference: "注册表引用",
    noMatchingRegistry: "没有匹配的注册项",
    customExtensionContract: "自定义扩展契约",
    registryFirstIntegration: "注册表优先的集成方式",
    extensionDescription:
      "先添加配置、适配器或训练器、公开测试样例、组合测试和一次有界真实验证，再升级兼容性证据。",
    configStep: "配置",
    adapterStep: "适配器",
    composeStep: "组合",
    smokeStep: "快速验证",
    evidenceStep: "证据",
    categoryFoundation: "基础",
    categoryExtension: "扩展",
    categoryMultimodal: "多模态",
    categoryAnalysis: "分析",
    categoryExecution: "执行",
    categoryData: "数据",
    categoryUi: "界面",
    categoryReporting: "报告",
    categoryRelease: "发布",
    categoryReproducibility: "可复现性",
    categoryValidation: "验证",
    uploadWorkflow: "上传流程",
    knowledgeType: "知识类型",
    uploadHint: "UTF-8 · JSON / JSONL / CSV · 最大 50 MiB",
    allRecordsValidated: "创建可运行配置前会校验全部记录。",
    fact: "事实",
    event: "事件",
    procedural: "过程",
    rule: "规则",
    freeText: "自由文本",
    concept: "概念",
    higherBetter: "越高越好",
    lowerBetter: "越低越好",
    closerZeroBetter: "越接近零越好",
    contextOnly: "仅供参考",
    metricOverview: "指标总览图",
    utilization: "利用率",
    requestFailed: "请求失败，请检查配置和服务端日志。"
  }
} as const;

const skillDescriptionsZh: Record<string, string> = {
  "add-dataset":
    "按照规范知识数据契约添加或校验数据集，并保留来源和许可证信息。",
  "add-model":
    "通过统一 Hydra 与适配器契约添加或校验模型；模型可加载不代表算法兼容。",
  "add-multimodal-case":
    "校验并注册图文知识用例及其探针，缺失或不可读的图像会直接报错。",
  "aggregate-results":
    "校验并聚合已完成运行的机器可读指标，不从日志猜测指标，也不以零填补缺失值。",
  "batch-sweep":
    "将已校验的手术计划展开为兼容的方法与随机种子运行清单。",
  "build-cases":
    "从规范化知识记录构造特定操作的运行用例和副作用评估探针。",
  "compare-methods":
    "在严格公平性契约下比较已完成方法，并准备排行榜和图表数据。",
  "debug-ui":
    "检查界面健康状态、命令等价性、交互状态和结果渲染。",
  "generate-report":
    "根据已校验计划、轨迹和聚合结果生成 Markdown 与 JSON 报告。",
  "inspect-repo":
    "审计仓库入口、注册配置、运行配方、技能和能力证据。",
  "normalize-knowledge":
    "校验 TXT、JSON、JSONL 或 CSV，并转换为规范知识记录格式。",
  "plan-surgery":
    "将自然语言知识更新请求转换为类型明确、证据感知的手术计划。",
  "release-checklist":
    "审计代码、测试、许可证、数据格式、论文声明、演示链接和视频状态。",
  "reproduce-run":
    "从运行产物重建完整命令、解析后配置和环境引用。",
  "run-edit":
    "通过统一入口预览并执行已注册的知识编辑计划。",
  "run-inject":
    "通过统一入口预览并执行已注册的知识注入计划。",
  "run-unlearn":
    "通过统一入口预览并执行已注册的知识遗忘计划，并保留副作用评估。",
  "setup-env":
    "检查并记录 Python、CUDA、GPU、软件包和本地模型缓存环境。",
  "smoke-test":
    "执行有界 Hydra 组合和可选真实模型快速验证，不将验证指标作为基准结果。"
};

export function displayTechnicalText(value: string): string {
  return value
    .replace(/smoke-test/gi, "quick-validation")
    .replace(/verified_smoke/gi, "verified_run")
    .replace(/smokes/gi, "validations")
    .replace(/smoke/gi, "validation")
    .replaceAll("冒烟测试", "快速验证")
    .replaceAll("冒烟", "验证");
}

export function localizedSkillDescription(
  id: string,
  language: Language,
  fallback: string
): string {
  return displayTechnicalText(
    language === "zh" ? skillDescriptionsZh[id] || fallback : fallback
  );
}

export function uiErrorMessage(error: unknown, language: Language): string {
  const raw = error instanceof Error ? error.message : String(error);
  if (language === "en") return displayTechnicalText(raw);
  if (/failed to fetch|network/i.test(raw)) {
    return "无法连接后端服务，请检查服务状态和端口。";
  }
  if (/not configured|api key/i.test(raw)) {
    return "所选规划器尚未在服务端配置。";
  }
  if (/gpu .*active|gpu .*busy|already has/i.test(raw)) {
    return "所选 GPU 已有活动任务，请选择其他 GPU。";
  }
  if (/already exists|choose a new/i.test(raw)) {
    return "该标识已经存在，请使用新的稳定标识。";
  }
  if (/upload|record|mapping|dataset/i.test(raw)) {
    return "数据校验失败，请检查文件格式、字段映射和数据集设置。";
  }
  if (/hydra|compose|experiment|trainer|model/i.test(raw)) {
    return "配置组合失败，请检查实验、模型、方法和 Hydra 配置。";
  }
  return messages.zh.requestFailed;
}

type MessageKey = keyof typeof messages.en;

interface I18nContextValue {
  language: Language;
  setLanguage: (value: Language) => void;
  toggleLanguage: () => void;
  t: (key: MessageKey) => string;
}

const I18nContext = createContext<I18nContextValue | null>(null);

export function I18nProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<Language>(() => {
    const stored = localStorage.getItem("know-surgery-language");
    return stored === "zh" ? "zh" : "en";
  });
  const update = useCallback((value: Language) => {
    localStorage.setItem("know-surgery-language", value);
    setLanguage(value);
  }, []);
  useEffect(() => {
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  }, [language]);
  const value = useMemo<I18nContextValue>(
    () => ({
      language,
      setLanguage: update,
      toggleLanguage: () => update(language === "en" ? "zh" : "en"),
      t: (key) => messages[language][key]
    }),
    [language, update]
  );
  return (
    <I18nContext.Provider value={value}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n(): I18nContextValue {
  const value = useContext(I18nContext);
  if (!value) {
    throw new Error("useI18n must be used within I18nProvider");
  }
  return value;
}
