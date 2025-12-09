import { routeAgentRequest, type Schedule } from "agents";
import { getSchedulePrompt } from "agents/schedule";

import { AIChatAgent } from "agents/ai-chat-agent";
import {
  generateId,
  streamText,
  type StreamTextOnFinishCallback,
  stepCountIs,
  createUIMessageStream,
  convertToModelMessages,
  createUIMessageStreamResponse
} from "ai";

import { openai } from "@ai-sdk/openai";
import { processToolCalls, cleanupMessages } from "./utils";
import { tools, executions } from "./tools";

const model = openai("gpt-4o-2024-11-20");

export interface Env {
  Chat: DurableObjectNamespace;
  DB: D1Database; // D1 binding from wrangler.jsonc
}

export class Chat extends AIChatAgent<Env> {
  async onChatMessage(
    onFinish: StreamTextOnFinishCallback<typeof tools>,
    _options?: { abortSignal?: AbortSignal }
  ) {
    const allTools = tools;

    const stream = createUIMessageStream({
      // 👇 lets the UI know which tools exist and which can be run
      tools: allTools,
      executions,

      execute: async ({ writer }) => {
        // 1) clean up any partial tool messages
        const cleanedMessages = cleanupMessages(this.messages);

        // 2) handle pending confirmation-required tools
        const processedMessages = await processToolCalls({
          messages: cleanedMessages,
          dataStream: writer,
          tools: allTools,
          executions,
          env: this.env
        });

        // 3) call the OpenAI model with tools
        const result = streamText({
          system: `You are a Treasury bond homework assistant running inside a Cloudflare Worker.

You have access to tools, including:

- "calculateTreasuryAnalytics": given a CUSIP, look up that security in the Cloudflare D1 database tables "nov18_treasury_securities" and "additional_data_new". Use the data to compute:
  - clean price
  - accrued interest
  - dirty price
  - the time-offset "f" (days into period / days in period)
  - plus day-count details (last and next coupon dates).

Whenever the user asks ANYTHING about:
- U.S. Treasury securities,
- CUSIPs,
- bond prices,
- clean price, dirty price, accrued interest, coupon dates, or "f",

you MUST call the "calculateTreasuryAnalytics" tool with the CUSIP INSTEAD of answering from your own knowledge. Only explain the results after the tool has run.

${getSchedulePrompt({ date: new Date() })}

If the user asks to schedule a task, use the scheduleTask tool to schedule the task.
`,
          messages: convertToModelMessages(processedMessages),
          model,
          tools: allTools,
          onFinish,
          stopWhen: stepCountIs(10)
        });

        writer.merge(result.toUIMessageStream());
      }
    });

    return createUIMessageStreamResponse({ stream });
  }

  async executeTask(description: string, _task: Schedule<string>) {
    await this.saveMessages([
      ...this.messages,
      {
        id: generateId(),
        role: "user",
        parts: [
          {
            type: "text",
            text: `Running scheduled task: ${description}`
          }
        ],
        metadata: {
          createdAt: new Date()
        }
      }
    ]);
  }
}

export default {
  async fetch(request: Request, env: Env, _ctx: ExecutionContext) {
    const url = new URL(request.url);

    if (url.pathname === "/check-open-ai-key") {
      const hasOpenAIKey = !!process.env.OPENAI_API_KEY;
      return Response.json({ success: hasOpenAIKey });
    }

    if (!process.env.OPENAI_API_KEY) {
      console.error(
        "OPENAI_API_KEY is not set, don't forget to set it locally in .dev.vars, and use `wrangler secret bulk .dev.vars` to upload it to production"
      );
    }

    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
