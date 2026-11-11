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
    connectionLost: "Connection unavailable",
    agentFirst: "Agent-first knowledge surgery",
    planIntent: "Describe the knowledge change",
    intentPlaceholder:
      "Example: Edit 2 CounterFact facts on Qwen2.5-1.5B with ROME, then evaluate locality.",
    generatePlan: "Generate plan",
    planning: "Planning",
    operation: "Operation",
    edit: "Editing",
    inject: "Injection",
    unlearning: "Unlearning",
    recipe: "Recipe",
    model: "Model",
    method: "Method",
    dataset: "Dataset",
    evaluation: "Evaluation",
    taskName: "Task name",
    limitEdits: "Limit to 1 edit",
    limitSteps: "Limit to 1 training step",
    seed: "Seed",
    preview: "Preview configuration",
    runNow: "Run task",
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
    realDemos: "Example results",
    metrics: "Metrics",
    process: "Process trace",
    selectResults: "Select 2-12 results",
    compareSelected: "Compare selected",
    comparable: "Comparable protocol",
    exploratory: "Side-by-side results",
    skills: "Agent Skills",
    catalog: "Capability catalog",
    configured: "Configured",
    local_validation: "Local run",
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
      "Choose a model, method, dataset, and evaluation. Required model mappings, checkpoints, statistics, and dependencies follow the selected method.",
    agentRecipe: "Agent",
    hide: "Hide",
    valid: "Valid",
    invalid: "Invalid",
    responseComparison: "MODEL RESPONSES",
    beforeAfterResponse: "Before / after model response",
    probeQuestion: "Probe question",
    expectedAnswer: "Expected answer",
    beforeUpdate: "Before update",
    afterUpdate: "After update",
    auditedArtifacts: "Result sources",
    missingArtifacts: "Source artifacts unavailable",
    customDataHint:
      "Convert text knowledge to JSON, JSONL, or CSV, review it, then compose.",
    customInjectionDataHint:
      "Map prompt and target_new, review every normalized record, then register a runnable injection recipe.",
    customEditingDataHint:
      "Map prompt, subject, target_new, and optional probes, then register a runnable editing recipe.",
    customUnlearningDataHint:
      "Upload separate forget and retain files, review both datasets, then register an unlearning recipe.",
    intentPresets: "Intent presets",
    longTextInjectionPreset: "Long-text injection · 7B · LoReFT",
    longTextEditingPreset: "Long-text editing · 7B · AnyEdit",
    availableSkills: "Shared project Skills",
    selectedForPlan: "suggested steps",
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
    auditTrace: "Execution record",
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
    compareEyebrow: "ANALYSIS / COMPARISON",
    compareDescription:
      "Compare model, input and evaluation settings before interpreting results.",
    fairnessMatch: "The recorded comparison settings match.",
    fairnessMismatch:
      "Some comparison settings are missing or differ. View these results side by side without ranking.",
    match: "Match",
    differs: "Differs",
    metric: "Metric",
    run: "Run",
    noCommonMetrics:
      "No common machine-readable metric exists across these runs.",
    fairnessBeforeRanking:
      "Select runs to compare their settings and measured effects.",
    fairnessOperation: "Operation",
    fairnessModel: "Model",
    fairnessDataset: "Dataset",
    fairnessScope: "Scope",
    fairnessEvidence: "Evidence level",
    extensionsEyebrow: "REGISTRIES / CONTRACTS",
    extensionsDescription:
      "Browse registered models, datasets, methods, and Agent Skills with their configuration details.",
    filterRegistry: "Filter registry",
    executableWrapper: "Executable wrapper",
    contractOnly: "Contract only",
    modelId: "Model ID",
    modality: "Modality",
    weightSource: "Weight source",
    cache: "Source type",
    status: "Status",
    datasetId: "Dataset ID",
    handlers: "Handlers",
    operations: "Operations",
    methodId: "Method ID",
    contract: "Contract",
    textModality: "Text",
    multimodalModality: "Multimodal",
    processorPathContract: "Weight source not listed",
    localPath: "Local path in config",
    registryReference: "Model repository ID",
    noMatchingRegistry: "No matching registry item",
    customExtensionContract: "CUSTOM EXTENSION CONTRACT",
    registryFirstIntegration: "Registry-first integration",
    extensionDescription:
      "Add a configuration and adapter or trainer, compose the configuration, then run the method and record its results.",
    configStep: "Config",
    adapterStep: "Adapter",
    composeStep: "Compose",
    smokeStep: "Run",
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
    connectionLost: "连接暂不可用",
    agentFirst: "智能体优先的知识手术",
    planIntent: "描述你的知识更新目标",
    intentPlaceholder:
      "示例：使用 ROME 在 Qwen2.5-1.5B 上编辑 2 条 CounterFact 事实，并评估局部性。",
    generatePlan: "生成计划",
    planning: "规划中",
    operation: "操作",
    edit: "编辑",
    inject: "注入",
    unlearning: "遗忘",
    recipe: "运行配方",
    model: "模型",
    method: "方法",
    dataset: "数据集",
    evaluation: "评估",
    taskName: "任务名",
    limitEdits: "限制为 1 条编辑",
    limitSteps: "限制为 1 个训练步",
    seed: "随机种子",
    preview: "组合配置",
    runNow: "执行任务",
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
    realDemos: "示例结果",
    metrics: "指标",
    process: "过程轨迹",
    selectResults: "选择 2-12 个结果",
    compareSelected: "对比所选结果",
    comparable: "协议可比",
    exploratory: "暂不排名",
    skills: "智能体技能",
    catalog: "能力目录",
    configured: "已配置",
    local_validation: "本地运行",
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
      "选择模型、方法、数据集和评估配置。模型映射、checkpoint、统计量和依赖按所选方法准备。",
    agentRecipe: "智能体",
    hide: "收起",
    valid: "有效",
    invalid: "无效",
    responseComparison: "模型回答",
    beforeAfterResponse: "模型更新前后回答",
    probeQuestion: "探针问题",
    expectedAnswer: "期望答案",
    beforeUpdate: "更新前",
    afterUpdate: "更新后",
    auditedArtifacts: "结果来源",
    missingArtifacts: "来源产物不可用",
    customDataHint:
      "请先将文本知识转换为 JSON、JSONL 或 CSV，审核后再组合配置。",
    customInjectionDataHint:
      "映射 prompt 与 target_new，审核全部规范化记录后，注册可运行的注入配方。",
    customEditingDataHint:
      "映射 prompt、subject、target_new 与可选探针后，注册可运行的编辑配方。",
    customUnlearningDataHint:
      "分别上传遗忘集与保留集，确认两份数据后，注册遗忘配方。",
    intentPresets: "目标模板",
    longTextInjectionPreset: "长文本注入 · 7B · LoReFT",
    longTextEditingPreset: "长文本编辑 · 7B · AnyEdit",
    availableSkills: "项目技能库",
    selectedForPlan: "项建议步骤",
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
    auditTrace: "执行记录",
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
    compareEyebrow: "分析 / 结果对比",
    compareDescription:
      "结合模型、输入和评测设置查看结果。",
    fairnessMatch: "已记录的对比设置一致。",
    fairnessMismatch: "部分比较条件缺失或不同，可并列查看结果，暂不排名。",
    match: "一致",
    differs: "不同",
    metric: "指标",
    run: "运行",
    noCommonMetrics: "这些运行之间没有共同的机器可读指标。",
    fairnessBeforeRanking: "选择运行，查看配置条件与实际效果。",
    fairnessOperation: "操作",
    fairnessModel: "模型",
    fairnessDataset: "数据集",
    fairnessScope: "数据范围",
    fairnessEvidence: "证据等级",
    extensionsEyebrow: "注册表 / 接口契约",
    extensionsDescription:
      "浏览已注册的模型、数据集、方法和智能体技能，查看各项配置。",
    filterRegistry: "筛选注册项",
    executableWrapper: "可执行封装",
    contractOnly: "仅接口契约",
    modelId: "模型 ID",
    modality: "模态",
    weightSource: "权重来源",
    cache: "来源类型",
    status: "状态",
    datasetId: "数据集 ID",
    handlers: "处理器",
    operations: "操作",
    methodId: "方法 ID",
    contract: "接口契约",
    textModality: "文本",
    multimodalModality: "多模态",
    processorPathContract: "目录未提供权重来源",
    localPath: "配置中的本地路径",
    registryReference: "模型仓库 ID",
    noMatchingRegistry: "没有匹配的注册项",
    customExtensionContract: "自定义扩展契约",
    registryFirstIntegration: "注册表优先的集成方式",
    extensionDescription:
      "添加配置与适配器或训练器，组合配置后执行方法并记录结果。",
    configStep: "配置",
    adapterStep: "适配器",
    composeStep: "组合",
    smokeStep: "运行",
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

const skillNamesZh: Record<string, string> = {
  "add-dataset": "添加数据集",
  "add-model": "添加模型",
  "add-multimodal-case": "添加图文样本",
  "aggregate-results": "汇总结果",
  "batch-sweep": "批量运行",
  "build-cases": "构造样本",
  "compare-methods": "比较方法",
  "debug-ui": "检查界面",
  "generate-report": "生成报告",
  "inspect-repo": "检查项目",
  "normalize-knowledge": "整理数据",
  "plan-surgery": "拟定计划",
  "release-checklist": "检查发布配置",
  "reproduce-run": "提取复现命令",
  "run-edit": "执行编辑",
  "run-inject": "执行注入",
  "run-unlearn": "执行遗忘",
  "setup-env": "检查环境",
  "smoke-test": "检查配置与输入"
};

export function localizedSkillName(id: string, language: Language): string {
  return language === "zh" ? skillNamesZh[id] || displayTechnicalText(id) : displayTechnicalText(id);
}

const skillDescriptionsZh: Record<string, string> = {
  "add-dataset":
    "按照规范知识数据契约添加或校验数据集，并保留来源和许可证信息。",
  "add-model":
    "添加模型配置与适配器，并检查所选方法需要的层映射和接口。",
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
    "查看仓库入口、注册配置、运行配方和技能。",
  "normalize-knowledge":
    "校验 TXT、JSON、JSONL 或 CSV，并转换为规范知识记录格式。",
  "plan-surgery":
    "将自然语言知识更新请求转换为包含模型、方法、数据和评估设置的执行计划。",
  "release-checklist":
    "检查代码入口、依赖、许可证、数据格式和发布资料。",
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
    "组合运行配置，并按计划使用所选模型处理最多两条记录。"
};

export function displayTechnicalText(value: string): string {
  return value
    .replace(/smoke[-_]test/gi, "run-check")
    .replace(/verified_smoke/gi, "verified_run")
    .replace(/smokes/gi, "runs")
    .replace(/smoke/gi, "run")
    .replaceAll("冒烟测试", "运行检查")
    .replaceAll("冒烟", "运行检查");
}

const resultReasons: Record<string, { en: string; zh: string }> = {
  before_not_recorded: { en: "The model response before the update was not recorded.", zh: "未记录更新前的模型回答。" },
  response_not_recorded: { en: "The model response was not recorded.", zh: "未记录模型回答。" },
  responses_not_recorded: { en: "The model responses before and after the update were not recorded.", zh: "未记录更新前后的模型回答。" },
  after_not_recorded: { en: "The model response after the update was not recorded.", zh: "未记录更新后的模型回答。" },
  missing_component_metrics: { en: "The overall score requires component metrics that are missing.", zh: "缺少计算综合得分所需的分项指标。" },
  missing_portability_inputs: { en: "No transfer questions were provided for evaluation.", zh: "未提供用于评测迁移性的问答。" },
  missing_locality_inputs: { en: "No locality questions were provided for evaluation.", zh: "未提供用于评测局部性的问答。" },
  missing_rephrase_prompts: { en: "No rephrased questions were provided for evaluation.", zh: "未提供用于评测泛化性的改写问题。" },
  missing_reliability_inputs: { en: "No target questions were provided for evaluation.", zh: "未提供用于评测目标效果的问答。" },
  semantic_model_not_configured: { en: "No model was configured to measure semantic similarity.", zh: "未配置用于计算语义相似度的模型。" },
  missing_original_model: { en: "The original model was not available for comparison.", zh: "未提供用于对照的原始模型。" },
  missing_pre_edit_baseline: { en: "No baseline before editing was recorded.", zh: "未记录编辑前的基线结果。" },
  missing_tokenizer: { en: "The tokenizer required for this evaluation was unavailable.", zh: "缺少此项评测所需的分词器。" },
  missing_processor: { en: "The image-text processor required for this evaluation was unavailable.", zh: "缺少此项评测所需的图文处理器。" },
  missing_tokenizer_or_edit_data: { en: "The tokenizer or editing data required for evaluation was unavailable.", zh: "缺少评测所需的分词器或编辑数据。" },
  missing_long_form_pairs: { en: "No long-form answers and references were available for comparison.", zh: "没有可用于比较的长文本回答与参考答案。" },
  missing_mmlu_probes: { en: "No MMLU questions were provided for evaluation.", zh: "未提供 MMLU 评测问题。" },
  no_valid_eval_samples: { en: "No valid evaluation samples were available.", zh: "没有可用的评测样本。" },
  no_valid_benchmark_samples: { en: "No valid benchmark samples were available.", zh: "没有可用的基准评测样本。" },
  no_valid_reference_samples: { en: "No valid reference samples were available.", zh: "没有可用的对照评测样本。" },
  no_evaluable_response_tokens: { en: "No answer tokens were available for scoring.", zh: "没有可用于评分的回答词元。" },
  original_score_non_positive: { en: "The original score is zero or negative, so a retention ratio cannot be calculated.", zh: "原始得分为零或负数，无法计算保留比例。" }
};

export function resultReasonText(reason: string, language: Language): string {
  return resultReasons[reason]?.[language] || (language === "zh"
    ? "此项记录不完整，可展开详情查看原始说明。"
    : "This item has incomplete information. Expand the details to see the recorded reason.");
}

export function localizedSkillDescription(
  id: string,
  language: Language,
  fallback: string
): string {
  return displayTechnicalText(
    language === "zh"
      ? skillDescriptionsZh[id] || fallback
      : id === "smoke-test"
        ? "Compose the configuration and use the selected model to process up to two records."
        : fallback
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
