export type Operation = "unlearning" | "inject" | "edit";
export type EvidenceLevel =
  | "planned"
  | "configured"
  | "local_validation"
  | "verified_smoke"
  | "verified_result";

export interface LocalizedText {
  en: string;
  zh: string;
}

export interface Catalog {
  schema_version: string;
  operations: Array<{
    id: Operation;
    label: LocalizedText;
    accent: string;
    metrics: string[];
  }>;
  models: Array<{
    id: string;
    modality: "text" | "multimodal";
    source?: string | null;
    local: boolean;
    status: EvidenceLevel;
  }>;
  trainers: Record<Operation, Array<{ id: string; status: EvidenceLevel }>>;
  datasets: Array<{
    id: string;
    handlers: string[];
    operations: Operation[];
    status: EvidenceLevel;
  }>;
  evaluations: string[];
  experiments: Array<{
    id: string;
    operation: Operation;
    status: EvidenceLevel;
  }>;
  recipes: Recipe[];
}

export interface Recipe {
  id: string;
  operation: Operation;
  experiment: string;
  model: string;
  trainer: string;
  datasets: Record<string, string>;
  evaluation: string;
  model_family?: string;
  knowledge_types: string[];
  status: EvidenceLevel;
  default_runnable: boolean;
}

export interface Plan {
  plan_id: string;
  schema_version: string;
  intent: string;
  operation: Operation;
  knowledge_type: string;
  model_id: string;
  dataset_id: string;
  methods: string[];
  num_cases: number;
  evaluators: string[];
  experiments: Record<string, string | null>;
  steps: Array<{
    skill: string;
    status: "pending" | "ready" | "blocked" | "completed";
    reason?: string;
  }>;
  warnings: string[];
  requires_confirmation: boolean;
  provider: string;
  reasoning: Array<{ field: string; reason: string }>;
}

export interface ComposePayload {
  operation: Operation;
  recipe_id?: string;
  experiment?: string;
  model?: string;
  trainer?: string;
  datasets?: Record<string, string>;
  evaluation?: string;
  task_name: string;
  seed: number;
  smoke: boolean;
  overrides?: Record<string, unknown>;
}

export interface ComposeResult {
  valid: boolean;
  returncode: number;
  command: string;
  preview_command: string;
  resolved_config: string;
  stderr: string;
  request: ComposePayload;
}

export interface Gpu {
  index: number;
  name: string;
  memory_total_mib: number;
  memory_used_mib: number;
  memory_free_mib: number;
  utilization_percent: number;
  available: boolean;
}

export interface Health {
  status: string;
  service: string;
  api_version: string;
  time: string;
  python: string;
  revision?: string;
  gpus: Gpu[];
  gpu_error?: string;
  skills: number;
  frontend_built: boolean;
  operations: Operation[];
}

export interface RunRecord {
  id: string;
  status:
    | "queued"
    | "running"
    | "stopping"
    | "succeeded"
    | "failed"
    | "stopped"
    | "interrupted";
  operation: Operation;
  task_name: string;
  gpu: number | null;
  created_at: string;
  started_at?: string;
  finished_at?: string;
  exit_code?: number | null;
  error?: string | null;
  command: string;
  output_dir: string;
  evidence_level: EvidenceLevel;
  event_count: number;
}

export interface RunEvent {
  index: number;
  timestamp: string;
  type: "status" | "log";
  status?: string;
  message?: string;
  stream?: string;
  exit_code?: number;
  evidence_level?: EvidenceLevel;
}

export interface Metric {
  id: string;
  value: number | null;
  source: string;
  label: LocalizedText;
  direction: "higher" | "lower" | "closer_to_zero" | "context";
  category: "efficacy" | "side_effect" | "efficiency" | "diagnostic";
  unit: string;
}

export interface ResultSummary {
  id: string;
  name: string;
  path: string;
  operation: Operation | "evaluation" | "unknown";
  evidence_level: EvidenceLevel;
  status: string;
  updated_at: number;
  metadata: Record<string, unknown>;
  metrics: Metric[];
}

export interface ResultDetail extends ResultSummary {
  timeline: Array<{
    id: string;
    label: LocalizedText;
    status: "completed" | "missing" | "failed" | "unknown";
  }>;
  files: string[];
  command?: string | null;
}

export interface Demo {
  slug: string;
  path: string;
  title: LocalizedText;
  summary: LocalizedText;
  result: ResultDetail;
}

export interface Skill {
  id: string;
  description: string;
  category: string;
  status: EvidenceLevel;
  has_runtime: boolean;
}

export interface ResponseExample {
  id: string;
  operation: Operation;
  evidence_level: EvidenceLevel;
  model: string;
  method: string;
  dataset: string;
  prompt: string;
  expected: string;
  before_response: string;
  after_response: string;
  outcome: LocalizedText;
  source_artifacts: string[];
  sources_available: boolean;
  decoding: Record<string, unknown>;
}
