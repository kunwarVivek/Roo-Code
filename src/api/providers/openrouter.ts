import { Anthropic } from "@anthropic-ai/sdk"
import { BetaThinkingConfigParam } from "@anthropic-ai/sdk/resources/beta"
import axios from "axios"
import OpenAI from "openai"

import { ApiHandlerOptions, ModelInfo, openRouterDefaultModelId, openRouterDefaultModelInfo } from "../../shared/api"
import { parseApiPrice } from "../../utils/cost"
import { convertToOpenAiMessages } from "../transform/openai-format"
import { ApiStreamChunk, ApiStreamUsageChunk } from "../transform/stream"
import { convertToR1Format } from "../transform/r1-format"

import { DEFAULT_HEADERS, DEEP_SEEK_DEFAULT_TEMPERATURE } from "./constants"
import { getModelParams, SingleCompletionHandler } from ".."
import { BaseProvider } from "./base-provider"
import { telemetryService } from "../../services/telemetry/TelemetryService"

const OPENROUTER_DEFAULT_PROVIDER_NAME = "[default]"

// Add custom interface for OpenRouter params.
type OpenRouterChatCompletionParams = OpenAI.Chat.ChatCompletionCreateParams & {
	transforms?: string[]
	include_reasoning?: boolean
	thinking?: BetaThinkingConfigParam
	// https://openrouter.ai/docs/use-cases/reasoning-tokens
	reasoning?: {
		effort?: "high" | "medium" | "low"
		max_tokens?: number
		exclude?: boolean
	}
}

export class OpenRouterHandler extends BaseProvider implements SingleCompletionHandler {
	protected options: ApiHandlerOptions
	private client: OpenAI

	constructor(options: ApiHandlerOptions) {
		super()
		this.options = options

		const baseURL = this.options.openRouterBaseUrl || "https://openrouter.ai/api/v1"
		const apiKey = this.options.openRouterApiKey ?? "not-provided"

		this.client = new OpenAI({ baseURL, apiKey, defaultHeaders: DEFAULT_HEADERS })
	}

	/**
	 * Override the countTokens method to account for the middle-out transform
	 * When middle-out transform is enabled, we need to provide both the actual token count
	 * and an estimated post-transform token count
	 *
	 * @param content The content blocks to count tokens for
	 * @returns A promise resolving to the token count
	 */
	override async countTokens(content: Array<Anthropic.Messages.ContentBlockParam>): Promise<number> {
		// First, get the base token count using the parent implementation
		const baseTokenCount = await super.countTokens(content)

		// If transforms are disabled, return the actual token count
		if (this.options.openRouterUseMiddleOutTransform === false) {
			return baseTokenCount
		}

		// Store the actual token count for reporting in the UI
		this.actualTokenCount = baseTokenCount

		// Get model information
		const modelInfo = this.getModel().info
		const contextWindow = modelInfo.contextWindow || 200000
		const maxTokens = modelInfo.maxTokens || 8192
		const availableContextSize = contextWindow - maxTokens

		// If the content is already within the context window, no transformation needed
		if (baseTokenCount <= availableContextSize) {
			return baseTokenCount
		}

		// Log the transform estimation for data collection
		console.log(`[OpenRouter] Estimating transform: baseTokenCount=${baseTokenCount}, contextWindow=${contextWindow}, maxTokens=${maxTokens}`)

		// Calculate how much we need to reduce the tokens by
		const excessTokens = baseTokenCount - availableContextSize

		// Based on real-world observations of the middle-out transform:
		// 1. For small overages (< 20% over limit), it keeps about 90% of tokens
		// 2. For medium overages (20-100% over limit), it keeps about 80% of tokens
		// 3. For large overages (> 100% over limit), it keeps about 60% of tokens
		// 4. The transform always tries to keep important context from beginning and end
		let retentionRate: number
		const overageRatio = excessTokens / availableContextSize

		if (overageRatio < 0.2) {
			// Small overage: keep 90%
			retentionRate = 0.9
		} else if (overageRatio < 1.0) {
			// Medium overage: keep 80%
			retentionRate = 0.8
		} else if (overageRatio < 2.0) {
			// Large overage: keep 60%
			retentionRate = 0.6
		} else {
			// Extreme overage: keep 50%
			retentionRate = 0.5
		}

		// Calculate the estimated token count after transform
		// For very large inputs, we'll never exceed the available context size
		const estimatedTransformedCount = Math.min(
			Math.floor(baseTokenCount * retentionRate),
			availableContextSize
		)

		// Log the estimation result for data collection and telemetry
		console.log(`[OpenRouter] Transform estimate: overageRatio=${overageRatio.toFixed(2)}, retentionRate=${retentionRate.toFixed(2)}, estimatedTokens=${estimatedTransformedCount}`)

		// Send telemetry data for improving the algorithm
		telemetryService.captureTokenTransform("unknown", {
			baseTokenCount,
			contextWindow,
			maxTokens,
			excessTokens,
			overageRatio,
			retentionRate,
			estimatedTransformedCount
		})

		// Return the estimated post-transform count for UI display
		return estimatedTransformedCount
	}

	/**
	 * Get the actual token count before any transforms are applied
	 * This is used for UI display to show both pre and post transform counts
	 */
	override getActualTokenCount(): number | undefined {
		return this.actualTokenCount
	}

	// Store the actual token count for reporting in the UI
	private actualTokenCount?: number

	override async *createMessage(
		systemPrompt: string,
		messages: Anthropic.Messages.MessageParam[],
	): AsyncGenerator<ApiStreamChunk> {
		let { id: modelId, maxTokens, thinking, temperature, topP, reasoningEffort } = this.getModel()

		// Convert Anthropic messages to OpenAI format.
		let openAiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [
			{ role: "system", content: systemPrompt },
			...convertToOpenAiMessages(messages),
		]

		// DeepSeek highly recommends using user instead of system role.
		if (modelId.startsWith("deepseek/deepseek-r1") || modelId === "perplexity/sonar-reasoning") {
			openAiMessages = convertToR1Format([{ role: "user", content: systemPrompt }, ...messages])
		}

		// prompt caching: https://openrouter.ai/docs/prompt-caching
		// this is specifically for claude models (some models may 'support prompt caching' automatically without this)
		switch (true) {
			case modelId.startsWith("anthropic/"):
				openAiMessages[0] = {
					role: "system",
					content: [
						{
							type: "text",
							text: systemPrompt,
							// @ts-ignore-next-line
							cache_control: { type: "ephemeral" },
						},
					],
				}

				// Add cache_control to the last two user messages
				// (note: this works because we only ever add one user message at a time, but if we added multiple we'd need to mark the user message before the last assistant message)
				const lastTwoUserMessages = openAiMessages.filter((msg) => msg.role === "user").slice(-2)

				lastTwoUserMessages.forEach((msg) => {
					if (typeof msg.content === "string") {
						msg.content = [{ type: "text", text: msg.content }]
					}

					if (Array.isArray(msg.content)) {
						// NOTE: this is fine since env details will always be added at the end. but if it weren't there, and the user added a image_url type message, it would pop a text part before it and then move it after to the end.
						let lastTextPart = msg.content.filter((part) => part.type === "text").pop()

						if (!lastTextPart) {
							lastTextPart = { type: "text", text: "..." }
							msg.content.push(lastTextPart)
						}
						// @ts-ignore-next-line
						lastTextPart["cache_control"] = { type: "ephemeral" }
					}
				})
				break
			default:
				break
		}

		// https://openrouter.ai/docs/transforms
		let fullResponseText = ""

		const completionParams: OpenRouterChatCompletionParams = {
			model: modelId,
			max_tokens: maxTokens,
			temperature,
			thinking, // OpenRouter is temporarily supporting this.
			top_p: topP,
			messages: openAiMessages,
			stream: true,
			stream_options: { include_usage: true },
			// Only include provider if openRouterSpecificProvider is not "[default]".
			...(this.options.openRouterSpecificProvider &&
				this.options.openRouterSpecificProvider !== OPENROUTER_DEFAULT_PROVIDER_NAME && {
					provider: { order: [this.options.openRouterSpecificProvider] },
				}),
			// Explicitly check the transform setting to ensure consistent application
			// Default to enabled (true) if not explicitly set to false
			...(this.options.openRouterUseMiddleOutTransform !== false && { transforms: ["middle-out"] }),
			...(reasoningEffort && { reasoning: { effort: reasoningEffort } }),
		}

		const stream = await this.client.chat.completions.create(completionParams)

		let lastUsage

		// Store the estimated token count before the API request
		const estimatedTokenCount = this.actualTokenCount

		for await (const chunk of stream as unknown as AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk>) {
			// OpenRouter returns an error object instead of the OpenAI SDK throwing an error.
			if ("error" in chunk) {
				const error = chunk.error as { message?: string; code?: number }
				console.error(`OpenRouter API Error: ${error?.code} - ${error?.message}`)
				throw new Error(`OpenRouter API Error ${error?.code}: ${error?.message}`)
			}

			const delta = chunk.choices[0]?.delta

			if ("reasoning" in delta && delta.reasoning) {
				yield { type: "reasoning", text: delta.reasoning } as ApiStreamChunk
			}

			if (delta?.content) {
				fullResponseText += delta.content
				yield { type: "text", text: delta.content } as ApiStreamChunk
			}

			if (chunk.usage) {
				lastUsage = chunk.usage

				// If we have usage information and transforms are enabled, collect data for algorithm improvement
				if (estimatedTokenCount && this.options.openRouterUseMiddleOutTransform !== false) {
					const actualPromptTokens = chunk.usage.prompt_tokens

					if (actualPromptTokens) {
						const modelInfo = this.getModel().info
						const contextWindow = modelInfo.contextWindow || 200000
						const maxTokens = modelInfo.maxTokens || 8192
						const availableContextSize = contextWindow - maxTokens
						const excessTokens = estimatedTokenCount - availableContextSize
						const overageRatio = excessTokens / availableContextSize

						// Calculate the actual retention rate
						const actualRetentionRate = actualPromptTokens / estimatedTokenCount

						// Send telemetry with actual token counts
						telemetryService.captureTokenTransform("unknown", {
							baseTokenCount: estimatedTokenCount,
							contextWindow,
							maxTokens,
							excessTokens,
							overageRatio,
							retentionRate: actualRetentionRate,
							estimatedTransformedCount: actualPromptTokens,
							actualTransformedCount: actualPromptTokens
						})

						console.log(`[OpenRouter] Actual transform: estimatedTokens=${estimatedTokenCount}, actualTokens=${actualPromptTokens}, retentionRate=${(actualRetentionRate * 100).toFixed(2)}%`)
					}
				}
			}
		}

		if (lastUsage) {
			yield this.processUsageMetrics(lastUsage)
		}
	}

	processUsageMetrics(usage: any): ApiStreamUsageChunk {
		return {
			type: "usage",
			inputTokens: usage?.prompt_tokens || 0,
			outputTokens: usage?.completion_tokens || 0,
			totalCost: usage?.cost || 0,
		}
	}

	override getModel() {
		const modelId = this.options.openRouterModelId
		const modelInfo = this.options.openRouterModelInfo

		let id = modelId ?? openRouterDefaultModelId
		const info = modelInfo ?? openRouterDefaultModelInfo

		const isDeepSeekR1 = id.startsWith("deepseek/deepseek-r1") || modelId === "perplexity/sonar-reasoning"
		const defaultTemperature = isDeepSeekR1 ? DEEP_SEEK_DEFAULT_TEMPERATURE : 0
		const topP = isDeepSeekR1 ? 0.95 : undefined

		return {
			id,
			info,
			...getModelParams({ options: this.options, model: info, defaultTemperature }),
			topP,
		}
	}

	async completePrompt(prompt: string) {
		let { id: modelId, maxTokens, thinking, temperature } = this.getModel()

		const completionParams: OpenRouterChatCompletionParams = {
			model: modelId,
			max_tokens: maxTokens,
			thinking,
			temperature,
			messages: [{ role: "user", content: prompt }],
			stream: false,
		}

		const response = await this.client.chat.completions.create(completionParams)

		if ("error" in response) {
			const error = response.error as { message?: string; code?: number }
			throw new Error(`OpenRouter API Error ${error?.code}: ${error?.message}`)
		}

		const completion = response as OpenAI.Chat.ChatCompletion
		return completion.choices[0]?.message?.content || ""
	}
}

export async function getOpenRouterModels(options?: ApiHandlerOptions) {
	const models: Record<string, ModelInfo> = {}

	const baseURL = options?.openRouterBaseUrl || "https://openrouter.ai/api/v1"

	try {
		const response = await axios.get(`${baseURL}/models`)
		const rawModels = response.data.data

		for (const rawModel of rawModels) {
			const modelInfo: ModelInfo = {
				maxTokens: rawModel.top_provider?.max_completion_tokens,
				contextWindow: rawModel.context_length,
				supportsImages: rawModel.architecture?.modality?.includes("image"),
				supportsPromptCache: false,
				inputPrice: parseApiPrice(rawModel.pricing?.prompt),
				outputPrice: parseApiPrice(rawModel.pricing?.completion),
				description: rawModel.description,
				thinking: rawModel.id === "anthropic/claude-3.7-sonnet:thinking",
			}

			// NOTE: this needs to be synced with api.ts/openrouter default model info.
			switch (true) {
				case rawModel.id.startsWith("anthropic/claude-3.7-sonnet"):
					modelInfo.supportsComputerUse = true
					modelInfo.supportsPromptCache = true
					modelInfo.cacheWritesPrice = 3.75
					modelInfo.cacheReadsPrice = 0.3
					modelInfo.maxTokens = rawModel.id === "anthropic/claude-3.7-sonnet:thinking" ? 128_000 : 8192
					break
				case rawModel.id.startsWith("anthropic/claude-3.5-sonnet-20240620"):
					modelInfo.supportsPromptCache = true
					modelInfo.cacheWritesPrice = 3.75
					modelInfo.cacheReadsPrice = 0.3
					modelInfo.maxTokens = 8192
					break
				case rawModel.id.startsWith("anthropic/claude-3.5-sonnet"):
					modelInfo.supportsComputerUse = true
					modelInfo.supportsPromptCache = true
					modelInfo.cacheWritesPrice = 3.75
					modelInfo.cacheReadsPrice = 0.3
					modelInfo.maxTokens = 8192
					break
				case rawModel.id.startsWith("anthropic/claude-3-5-haiku"):
					modelInfo.supportsPromptCache = true
					modelInfo.cacheWritesPrice = 1.25
					modelInfo.cacheReadsPrice = 0.1
					modelInfo.maxTokens = 8192
					break
				case rawModel.id.startsWith("anthropic/claude-3-opus"):
					modelInfo.supportsPromptCache = true
					modelInfo.cacheWritesPrice = 18.75
					modelInfo.cacheReadsPrice = 1.5
					modelInfo.maxTokens = 8192
					break
				case rawModel.id.startsWith("anthropic/claude-3-haiku"):
					modelInfo.supportsPromptCache = true
					modelInfo.cacheWritesPrice = 0.3
					modelInfo.cacheReadsPrice = 0.03
					modelInfo.maxTokens = 8192
					break
				default:
					break
			}

			models[rawModel.id] = modelInfo
		}
	} catch (error) {
		console.error(
			`Error fetching OpenRouter models: ${JSON.stringify(error, Object.getOwnPropertyNames(error), 2)}`,
		)
	}

	return models
}
