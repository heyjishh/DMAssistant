package com.aireply.keyboard.ai

/**
 * Provider presets that prefill the API settings. gemini-web2api is a self-hosted
 * server (github.com/Sophomoresty/gemini-web2api) exposing the Gemini web UI as an
 * OpenAI-compatible API — free, and the API key is optional (only needed if the
 * server's config.json defines api_keys).
 */
object ProviderPresets {

    data class Preset(
        val label: String,
        val baseUrl: String?,
        val textModel: String?,
        val visionModel: String?,
        val keyHint: String
    )

    val CUSTOM = Preset("Custom", null, null, null, "")

    val all = listOf(
        Preset(
            "gemini-web2api (self-hosted, free)",
            "http://10.0.2.2:8081/v1",   // Android emulator -> host machine
            "gemini-3.6-flash",
            "gemini-3.6-flash",
            "Optional — only if api_keys is set in the server's config.json"
        ),
        Preset(
            "OpenAI",
            "https://api.openai.com/v1",
            "gpt-4o-mini",
            "gpt-4o-mini",
            "Required"
        ),
        Preset(
            "Gemini API (official)",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "gemini-2.0-flash",
            "gemini-2.0-flash",
            "Required (Google AI Studio key)"
        ),
        Preset(
            "OpenRouter",
            "https://openrouter.ai/api/v1",
            "openai/gpt-4o-mini",
            "openai/gpt-4o-mini",
            "Required"
        ),
        CUSTOM
    )
}
