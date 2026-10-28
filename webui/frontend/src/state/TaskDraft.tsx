import { createContext, useCallback, useContext, useReducer, type ReactNode, type SetStateAction } from "react";
import type { ComposePayload, ComposeResult, Modality, Operation, Plan, RunRecord } from "../types";

export interface AgentTrace {
  session_id: string;
  status: string;
  response?: string;
  error?: string;
  run_id?: string;
  result_id?: string;
  run_status?: string;
  calls: { skill: string; status: string; command?: string; error?: string; output?: string }[];
}

interface TaskDraft {
  initialized: boolean;
  operation: Operation;
  modality: Modality;
  intent: string;
  provider: string;
  plan: Plan | null;
  recipeId: string;
  experiment: string;
  model: string;
  trainer: string;
  datasets: Record<string, string>;
  evaluation: string;
  taskName: string;
  seed: number;
  smoke: boolean;
  overrides: Record<string, unknown>;
  overridesDraft: string | null;
  baseSelections: Pick<ComposePayload, "model" | "trainer" | "datasets" | "evaluation">;
  composeSnapshot: { key: string; result: ComposeResult } | null;
  currentRun: RunRecord | null;
  agentTrace: AgentTrace | null;
}

const initialDraft: TaskDraft = {
  initialized: false, operation: "unlearning", modality: "text", intent: "", provider: "local",
  plan: null, recipeId: "", experiment: "", model: "", trainer: "", datasets: {}, evaluation: "",
  taskName: "", seed: 42, smoke: false, overrides: {}, overridesDraft: null, baseSelections: {}, composeSnapshot: null, currentRun: null, agentTrace: null
};
type DraftUpdate = (draft: TaskDraft) => TaskDraft;
const DraftContext = createContext<{ draft: TaskDraft; update: (update: DraftUpdate) => void } | null>(null);

/** One in-memory task across routes; reload deliberately requires a fresh preview. */
export function TaskDraftProvider({ children }: { children: ReactNode }) {
  const [draft, update] = useReducer((state: TaskDraft, action: DraftUpdate) => action(state), initialDraft);
  return <DraftContext.Provider value={{ draft, update }}>{children}</DraftContext.Provider>;
}

export function useDraftField<K extends keyof TaskDraft>(key: K): [TaskDraft[K], (value: SetStateAction<TaskDraft[K]>) => void] {
  const context = useContext(DraftContext);
  if (!context) throw new Error("TaskDraftProvider is required");
  const { draft, update } = context;
  const setValue = useCallback((value: SetStateAction<TaskDraft[K]>) => {
    update(old => {
      const next = typeof value === "function" ? (value as (previous: TaskDraft[K]) => TaskDraft[K])(old[key]) : value;
      return Object.is(next, old[key]) ? old : { ...old, [key]: next };
    });
  }, [key, update]);
  return [draft[key], setValue];
}
