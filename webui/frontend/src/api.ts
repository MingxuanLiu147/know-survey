import type {
  Catalog,
  ComposePayload,
  ComposeResult,
  Demo,
  Health,
  Operation,
  Plan,
  ResponseExample,
  ResultDetail,
  ResultSummary,
  RunEvent,
  RunRecord,
  Skill
} from "./types";

const API = "/api/v1";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const response = await fetch(API + path, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData
        ? {}
        : { "Content-Type": "application/json" }),
      ...init?.headers
    }
  });
  if (!response.ok) {
    let message = response.statusText;
    try {
      const body = await response.json();
      message =
        typeof body.detail === "string"
          ? body.detail
          : JSON.stringify(body.detail);
    } catch {
      message = await response.text();
    }
    throw new ApiError(message || "Request failed", response.status);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/health"),
  catalog: () => request<Catalog>("/catalog"),
  skills: () =>
    request<{ skills: Skill[]; count: number }>("/skills"),
  responseExamples: () =>
    request<{ examples: ResponseExample[]; count: number }>(
      "/response-examples"
    ),
  providers: () =>
    request<{
      providers: Array<{
        id: string;
        label: string;
        configured: boolean;
        key_source?: string;
      }>;
    }>("/providers"),
  plan: (payload: {
    intent: string;
    provider: string;
    language: "en" | "zh";
    operation?: string;
  }) =>
    request<Plan>("/plans", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  compose: (payload: ComposePayload) =>
    request<ComposeResult>("/compose", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  startRun: (payload: ComposePayload & { gpu: number | null }) =>
    request<RunRecord>("/runs", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  runs: () => request<{ runs: RunRecord[] }>("/runs"),
  run: (id: string) => request<RunRecord>("/runs/" + id),
  runEvents: (id: string, after = -1) =>
    request<{ events: RunEvent[] }>(
      "/runs/" + id + "/event-list?after=" + after
    ),
  stopRun: (id: string) =>
    request<RunRecord>("/runs/" + id + "/stop", {
      method: "POST"
    }),
  results: () =>
    request<{ results: ResultSummary[] }>("/results"),
  result: (id: string) =>
    request<ResultDetail>("/results/" + id),
  demos: () => request<{ demos: Demo[] }>("/demos"),
  compare: (ids: string[]) =>
    request<{
      comparable: boolean;
      fairness: Record<
        string,
        { values: unknown[]; known: boolean; equal: boolean }
      >;
      warnings: string[];
      metric_ids: string[];
      results: ResultDetail[];
    }>("/results/compare", {
      method: "POST",
      body: JSON.stringify({ result_ids: ids })
    }),
  previewUpload: (operation: Operation, form: FormData) =>
    request<{
      upload_id: string;
      records: number;
      preview: Array<Record<string, unknown>>;
      field_mapping: Record<string, string>;
      source_sha256: string;
    }>("/uploads/" + operation + "/preview", {
      method: "POST",
      body: form
    }),
  registerUpload: (
    operation: Operation,
    payload: {
    upload_id: string;
    dataset_id: string;
    model: string;
    trainer: string;
    evaluation: string;
    seed: number;
    }
  ) =>
    request<{
      dataset_id: string;
      experiment: string;
      command: string;
      compose_request: ComposePayload;
    }>("/uploads/" + operation + "/register", {
      method: "POST",
      body: JSON.stringify(payload)
    })
};
