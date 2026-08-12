import { describe, expect, test } from "bun:test";
import * as AIError from "@oh-my-pi/pi-ai/error";
import { streamBedrock } from "@oh-my-pi/pi-ai/providers/amazon-bedrock";
import { crc32 } from "@oh-my-pi/pi-ai/providers/aws-eventstream";
import type { Context, Model } from "@oh-my-pi/pi-ai/types";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import type { FetchImpl } from "@oh-my-pi/pi-catalog/types";

// ---- Minimal eventstream frame encoder (string headers only), mirroring the
// fixture builder in aws-eventstream.test.ts: the decoder under test is
// production code; encoding exists only to own the fixture bytes.

function encodeStringHeader(name: string, value: string): Uint8Array {
	const nameBytes = new TextEncoder().encode(name);
	const valueBytes = new TextEncoder().encode(value);
	const buf = new Uint8Array(1 + nameBytes.length + 1 + 2 + valueBytes.length);
	const view = new DataView(buf.buffer);
	let p = 0;
	view.setUint8(p, nameBytes.length);
	p += 1;
	buf.set(nameBytes, p);
	p += nameBytes.length;
	view.setUint8(p, 7); // string type
	p += 1;
	view.setUint16(p, valueBytes.length, false);
	p += 2;
	buf.set(valueBytes, p);
	return buf;
}

function encodeFrame(headers: Record<string, string>, payload: Uint8Array): Uint8Array {
	const headerChunks: Uint8Array[] = [];
	for (const name in headers) headerChunks.push(encodeStringHeader(name, headers[name]));
	const headerLen = headerChunks.reduce((s, c) => s + c.length, 0);
	const headerBytes = new Uint8Array(headerLen);
	let off = 0;
	for (const c of headerChunks) {
		headerBytes.set(c, off);
		off += c.length;
	}
	const total = 4 + 4 + 4 + headerLen + payload.length + 4;
	const out = new Uint8Array(total);
	const view = new DataView(out.buffer);
	view.setUint32(0, total, false);
	view.setUint32(4, headerLen, false);
	view.setUint32(8, crc32(out.subarray(0, 8)), false);
	out.set(headerBytes, 12);
	out.set(payload, 12 + headerLen);
	view.setUint32(total - 4, crc32(out.subarray(0, total - 4)), false);
	return out;
}

function eventFrame(eventType: string, payload: object): Uint8Array {
	return encodeFrame(
		{ ":message-type": "event", ":event-type": eventType, ":content-type": "application/json" },
		new TextEncoder().encode(JSON.stringify(payload)),
	);
}

function exceptionFrame(exceptionType: string, payload: object): Uint8Array {
	return encodeFrame(
		{ ":message-type": "exception", ":exception-type": exceptionType, ":content-type": "application/json" },
		new TextEncoder().encode(JSON.stringify(payload)),
	);
}

function bodyFrom(frames: Uint8Array[]): ReadableStream<Uint8Array> {
	let i = 0;
	return new ReadableStream({
		pull(controller) {
			if (i < frames.length) controller.enqueue(frames[i++]);
			else controller.close();
		},
	});
}

const context: Context = {
	systemPrompt: ["Use concise answers."],
	messages: [{ role: "user", content: "What is the answer?", timestamp: 0 }],
};

const model: Model<"bedrock-converse-stream"> = buildModel({
	id: "anthropic.claude-opus-4-6-v1",
	name: "test",
	api: "bedrock-converse-stream",
	provider: "amazon-bedrock",
	baseUrl: "https://bedrock-runtime.us-east-1.amazonaws.com",
	reasoning: false,
	input: ["text"],
	cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
	contextWindow: 1_000_000,
	maxTokens: 128_000,
});

async function failWithMidStreamException(exceptionType: string, message: string) {
	const frames = [eventFrame("messageStart", { role: "assistant" }), exceptionFrame(exceptionType, { message })];
	const fetchImpl: FetchImpl = async () =>
		new Response(bodyFrom(frames), {
			status: 200,
			headers: { "content-type": "application/vnd.amazon.eventstream" },
		});
	const stream = streamBedrock(model, context, { bearerToken: "test-token", fetch: fetchImpl });
	return await stream.result();
}

describe("Bedrock mid-stream exception status mapping", () => {
	test("internalServerException surfaces as a retryable 500, not a terminal 400", async () => {
		// AWS emits this transient server fault mid-stream ("Try your request
		// again.") after the HTTP 200 handshake, so no HTTP status exists — the
		// smithy exception type must map to 500 or auto-retry classifies the
		// turn as a terminal client error.
		const result = await failWithMidStreamException(
			"internalServerException",
			"The system encountered an unexpected error during processing. Try your request again.",
		);
		expect(result.stopReason).toBe("error");
		expect(result.errorStatus).toBe(500);
		expect(result.errorMessage).toContain("internalServerException");
		const id = AIError.classifyMessage(result);
		expect(AIError.is(id, AIError.Flag.Transient)).toBe(true);
		expect(AIError.retriable(id)).toBe(true);
	});

	test("throttlingException maps to 429 and stays retryable", async () => {
		const result = await failWithMidStreamException("throttlingException", "Too many requests, please slow down.");
		expect(result.stopReason).toBe("error");
		expect(result.errorStatus).toBe(429);
		expect(AIError.retriable(AIError.classifyMessage(result))).toBe(true);
	});

	test("validationException stays a terminal 400", async () => {
		const result = await failWithMidStreamException("validationException", "Input is malformed.");
		expect(result.stopReason).toBe("error");
		expect(result.errorStatus).toBe(400);
		expect(AIError.retriable(AIError.classifyMessage(result))).toBe(false);
	});

	test("modelStreamErrorException relays the upstream model status", async () => {
		const frames = [
			exceptionFrame("modelStreamErrorException", { message: "Upstream model error.", originalStatusCode: 503 }),
		];
		const fetchImpl: FetchImpl = async () =>
			new Response(bodyFrom(frames), {
				status: 200,
				headers: { "content-type": "application/vnd.amazon.eventstream" },
			});
		const stream = streamBedrock(model, context, { bearerToken: "test-token", fetch: fetchImpl });
		const result = await stream.result();
		expect(result.errorStatus).toBe(503);
		expect(AIError.retriable(AIError.classifyMessage(result))).toBe(true);
	});
});

describe("Bedrock content_filtered stop reason", () => {
	test("surfaces as a labeled refusal: stopDetails, content-blocked flag, non-retryable", async () => {
		const frames = [
			eventFrame("messageStart", { role: "assistant" }),
			eventFrame("messageStop", { stopReason: "content_filtered" }),
		];
		const fetchImpl: FetchImpl = async () =>
			new Response(bodyFrom(frames), {
				status: 200,
				headers: { "content-type": "application/vnd.amazon.eventstream" },
			});
		const stream = streamBedrock(model, context, { bearerToken: "test-token", fetch: fetchImpl });
		const result = await stream.result();
		expect(result.stopReason).toBe("error");
		expect(result.stopDetails?.type).toBe("refusal");
		expect(result.errorMessage).toMatch(/^Refusal/);
		expect(result.errorMessage).toContain("content_filtered");
		expect(AIError.is(result.errorId, AIError.Flag.ContentBlocked)).toBe(true);
		expect(AIError.retriable(result.errorId)).toBe(false);
	});
});
