import type { LanguageModel } from "ai";

export interface ToolSpec {
  id: string;
  name: string;
  description: string;
  inputSchema?: Record<string, unknown>;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface TurnInput {
  messages: Message[];
}

export interface Turn {
  input: TurnInput;
  expectedTool: string;
  expectedQuery: string;
}

export interface Scenario {
  id: string;
  turns: Turn[];
}

export interface Dataset {
  tools: ToolSpec[];
  scenarios: Scenario[];
}

export interface AgentRunResult {
  finalText: string;
  toolCalls: Array<{ toolId: string; args: unknown }>;
  effectiveToolIds: string[];
  tokens: {
    input: number;
    output: number;
    cachedInput: number;
    cacheCreation: number;
    total: number;
  };
  wallMs: number;
  finishReason: string;
  error: string | null;
}

export interface AgentInstance {
  run: (input: TurnInput) => Promise<AgentRunResult>;
}

export interface Agent {
  id: string;
  init: (args: { toolPool: ToolSpec[]; model: LanguageModel }) => AgentInstance;
}
