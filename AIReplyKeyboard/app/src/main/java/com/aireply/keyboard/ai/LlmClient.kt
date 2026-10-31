package com.aireply.keyboard.ai

import android.content.Context
import android.os.Handler
import android.os.Looper
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

object LlmClient {

    // Lazy so unit tests can use the pure functions without touching Android stubs.
    private val client: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(15, TimeUnit.SECONDS)
            .readTimeout(90, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .build()
    }

    private val main: Handler by lazy { Handler(Looper.getMainLooper()) }
    private val json = "application/json; charset=utf-8".toMediaType()

    private fun styleLine(context: Context): String {
        val style = Prefs.styleProfile(context)
        return if (style.isEmpty()) "" else
            " The user's personal texting style (adapt to it where possible): $style."
    }

    private fun systemPrompt(context: Context, tone: Tone): String {
        val bilingualLine = if (Prefs.bilingual(context)) {
            " Write the replies in natural Hinglish (Roman-script Hindi-English mix, like " +
                "'kya scene hai', 'chal na yaar', 'thik hai'). When describing HER actions use " +
                "feminine verb forms (kar rhi ho, karti ho, bna leti ho) — never masculine forms."
        } else ""
        return "You are an elite texting coach writing for the user (a guy) inside his keyboard. " +
            "The conversation is pasted below; its last line is the latest message from the other person. " +
            "Write exactly 3 candidate replies the user could send next, in the style of: " +
            "${tone.instruction}.${styleLine(context)}$bilingualLine\n" +
            "GAME RULES — apply silently, never mention them:\n" +
            "- Mirror their energy: a dry one-liner gets a short confident reply; excitement gets playful energy back\n" +
            "- Under 20 words each, punchy, ONE focused point per reply\n" +
            "- Specificity beats flattery: reference a detail they actually said or posted; no canned lines like 'hey beautiful'\n" +
            "- A light tease or playful challenge wins over compliments; if complimenting, praise taste/humor/energy — never looks early on\n" +
            "- Weave one small self-reveal across the 3 replies (a tiny detail about the user) to earn reciprocity\n" +
            "- Prefer fun open-ended hooks (why / how / favorite / would-you-rather) that are tempting to answer\n" +
            "- If there are 2+ back-and-forths, one reply confidently moves toward a concrete plan (specific day + place), low pressure\n" +
            "- If their energy is dry: at most ONE warm value-adding reply; make another a graceful light exit; NEVER chase, never double-question, never 'why no reply'\n" +
            "- If a warning is included below, obey it strictly\n" +
            "- Text like a real human: contractions, casual casing ok, max 1-2 emojis, no em dashes, no hashtags, no assistant-speak, no apologizing for texting\n" +
            "Output format: reply1 ||| reply2 ||| reply3 on one line, always 3, no numbering, no quotes."
    }

    /** Rewrites the user's own draft in the selected tone. */
    fun rewriteText(
        context: Context,
        tone: Tone,
        draft: String,
        onResult: (Result<String>) -> Unit
    ) {
        val sys = "Rewrite the user's draft message in the style of: ${tone.instruction}." +
            "${styleLine(context)} Keep the exact meaning and the original language, make it " +
            "natural for texting, and output ONLY the rewritten message with no quotes, " +
            "commentary or alternatives."
        val body = JSONObject().apply {
            put("model", Prefs.model(context))
            put("max_tokens", 200)
            put("temperature", 0.8)
            put("messages", JSONArray().apply {
                put(JSONObject().put("role", "system").put("content", sys))
                put(JSONObject().put("role", "user").put("content", "Draft:\n$draft"))
            })
        }
        postChat(context, body) { result ->
            main.post {
                onResult(result.mapCatching { it.trim().removeSurrounding("\"").ifEmpty { throw IOException("Empty rewrite.") } })
            }
        }
    }

    /** Modes for screenshot analysis. */
    enum class AnalysisTask(val instruction: String) {
        OPENERS(
            "Suggest exactly 3 openers the user could send — reference specific details from the " +
                "screenshot (bio, photos, prompts) rather than generic greetings. If the content is a " +
                "reel or feed post, suggest reaction comments instead."
        ),
        REPLIES(
            "The screenshot shows a conversation; the last message shown is the one to respond to. " +
                "Suggest exactly 3 replies the user could send next."
        )
    }

    fun generate(
        context: Context,
        tone: Tone,
        conversation: String,
        avoid: List<String> = emptyList(),
        onResult: (Result<List<String>>) -> Unit
    ) {
        val read = MoodSensor.read(conversation)
        val moodContext = "Context read: the other person's current mood seems \"$read.mood\". " +
            MoodSensor.moodAdvice(read.mood) +
            (read.trapAdvice?.let { " Warning: $it" } ?: "")
        val avoidLine = if (avoid.isEmpty()) "" else
            "\n\nThe user already saw these suggestions and tapped again — offer clearly different ones:\n" +
                avoid.joinToString("\n") { "- $it" }

        val body = JSONObject().apply {
            put("model", Prefs.model(context))
            put("max_tokens", 300)
            put("temperature", 0.9)
            put("messages", JSONArray().apply {
                put(JSONObject().put("role", "system").put("content", systemPrompt(context, tone)))
                put(
                    JSONObject().put("role", "user")
                        .put("content", "Conversation so far (oldest first, last line is the latest message):\n$conversation\n\n$moodContext$avoidLine")
                )
            })
        }
        postChat(context, body) { result -> result.fold(
            onSuccess = { r -> main.post { onResult(Result.success(parseReplies(r))) } },
            onFailure = { e -> main.post { onResult(Result.failure(e)) } }
        ) }
    }

    /** RIZZ-style: send a chat/bio screenshot to a vision model for openers or replies. */
    fun analyzeImage(
        context: Context,
        tone: Tone,
        task: AnalysisTask,
        imageBase64: String,
        onResult: (Result<Pair<String?, List<String>>>) -> Unit
    ) {
        val sys = "${task.instruction} Style: ${tone.instruction}.${styleLine(context)} " +
            "Match the language of the screenshot. Do not number the suggestions, do not add " +
            "quotes or commentary; separate the 3 suggestions with the exact delimiter |||." +
            " IMPORTANT: on the very first line output exactly one word 'TYPE:' followed by what the " +
            "screenshot shows — PROFILE, CHAT, REEL or INFEED — then the suggestions on the next lines."

        val detectionRegex = Regex("TYPE:\\s*(\\w+)", RegexOption.IGNORE_CASE)

        val body = JSONObject().apply {
            put("model", Prefs.visionModel(context))
            put("max_tokens", 400)
            put("temperature", 0.9)
            put("messages", JSONArray().apply {
                put(JSONObject().put("role", "system").put("content", sys))
                put(JSONObject().put("role", "user").put("content", JSONArray().apply {
                    put(JSONObject().put("type", "text")
                        .put("text", "Analyze this screenshot and reply with exactly 3 suggestions separated by ||| ."))
                    put(JSONObject().put("type", "image_url")
                        .put("image_url", JSONObject().put("url", "data:image/jpeg;base64,$imageBase64")))
                }))
            })
        }
        postChat(context, body) { result -> result.fold(
            onSuccess = { raw ->
                val detected = detectionRegex.find(raw)?.groupValues?.get(1)?.uppercase()
                val cleaned = raw.replace(detectionRegex, "").trim()
                main.post { onResult(Result.success(Pair(detected, parseReplies(cleaned)))) }
            },
            onFailure = { e -> main.post { onResult(Result.failure(e)) } }
        ) }
    }

    /** FlirtCopilot-style red/green flag audit of a conversation screenshot. */
    fun auditImage(
        context: Context,
        imageBase64: String,
        onResult: (Result<String>) -> Unit
    ) {
        val body = JSONObject().apply {
            put("model", Prefs.visionModel(context))
            put("max_tokens", 400)
            put("temperature", 0.4)
            put("messages", JSONArray().apply {
                put(JSONObject().put("role", "system").put("content", AUDIT_PROMPT))
                put(JSONObject().put("role", "user").put("content", JSONArray().apply {
                    put(JSONObject().put("type", "text")
                        .put("text", "Audit the conversation in this screenshot."))
                    put(JSONObject().put("type", "image_url")
                        .put("image_url", JSONObject().put("url", "data:image/jpeg;base64,$imageBase64")))
                }))
            })
        }
        postChat(context, body) { result -> main.post { onResult(result) } }
    }

    private val AUDIT_PROMPT =
        "You are a friendly communication coach. Analyze this conversation for signals " +
        "about how the other person is engaging. Reply in exactly this format:\n" +
        "GREEN FLAGS\n- up to 3 short bullets\n\nRED FLAGS\n- up to 3 short bullets\n\n" +
        "OVERALL\nOne short constructive sentence about the vibe and what to do next.\n" +
        "Be specific to the conversation, kind but honest. Never comment on anyone's " +
        "appearance. If there is too little text to judge, say so in OVERALL."

    /** Live-mode: user is browsing a social/dating app; analyze captured screen + visible text. */
    fun generateFromLiveScreen(
        context: Context,
        tone: Tone,
        imageBase64: String,
        screenText: String,
        avoid: List<String> = emptyList(),
        onResult: (Result<Pair<String?, List<String>>>) -> Unit
    ) {
        val sys = "You are a flirting/texting assistant with live screen access. The user is browsing a " +
            "social or dating app (Instagram feed/reel, Hinge, Tinder...). The screenshot shows what they " +
            "see — a profile, post or reel; ignore any on-screen keyboard UI at the bottom of the image. " +
            "On-screen text extracted separately may include bio, caption, username or prompts. " +
            "First line of output must be exactly: TYPE: PROFILE, CHAT, REEL or INFEED. " +
            "Then write exactly 3 short openers or replies the user could send, referencing specific " +
            "visible details (bio text, caption, image content) rather than generic greetings. " +
            "Style: ${tone.instruction}.${styleLine(context)} " +
            "Match the language of the screen. Do not number or quote; output exactly " +
            "reply1 ||| reply2 ||| reply3 on one line, always 3 even if short." +
            (if (avoid.isEmpty()) "" else " The user already saw these and tapped again — offer clearly different ones:\n" +
                avoid.joinToString("\n") { "- $it" })

        val read = MoodSensor.read(screenText)
        val moodContext = "Context read: mood \"${read.mood}\". " + MoodSensor.moodAdvice(read.mood) +
            (read.trapAdvice?.let { " Warning: $it" } ?: "")

        val body = JSONObject().apply {
            put("model", Prefs.visionModel(context))
            put("max_tokens", 400)
            put("temperature", 0.9)
            put("messages", JSONArray().apply {
                put(JSONObject().put("role", "system").put("content", sys))
                put(JSONObject().put("role", "user").put("content", JSONArray().apply {
                    put(JSONObject().put("type", "text")
                        .put("text", "On-screen text:\n${screenText.take(2000)}\n\n$moodContext"))
                    put(JSONObject().put("type", "image_url")
                        .put("image_url", JSONObject().put("url", "data:image/jpeg;base64,$imageBase64")))
                }))
            })
        }
        postChat(context, body) { result -> result.fold(
            onSuccess = { raw ->
                val detected = Regex("TYPE:\\s*(\\w+)", RegexOption.IGNORE_CASE).find(raw)?.groupValues?.get(1)?.uppercase()
                val cleaned = raw.replace(Regex("TYPE:\\s*\\w+", RegexOption.IGNORE_CASE), "").trim()
                main.post { onResult(Result.success(Pair(detected, parseReplies(cleaned)))) }
            },
            onFailure = { e -> main.post { onResult(Result.failure(e)) } }
        ) }
    }

    private fun postChat(context: Context, body: JSONObject, done: (Result<String>) -> Unit) {
        val builder = Request.Builder()
            .url("${Prefs.baseUrl(context)}/chat/completions")
        // Key is optional: local/self-hosted servers (e.g. gemini-web2api) may run without auth.
        val apiKey = Prefs.apiKey(context)
        if (apiKey.isNotEmpty()) builder.header("Authorization", "Bearer $apiKey")
        val request = builder.post(body.toString().toRequestBody(json)).build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                main.post { done(Result.failure(e)) }
            }

            override fun onResponse(call: Call, response: okhttp3.Response) {
                val result = runCatching {
                    val text = response.body?.string().orEmpty()
                    if (!response.isSuccessful) {
                        android.util.Log.e("LlmClient", "API error ${response.code}: ${text.take(500)}")
                        throw IOException("API error ${response.code}: ${text.take(300)}")
                    }
                    val content = JSONObject(text)
                        .getJSONArray("choices").getJSONObject(0)
                        .getJSONObject("message").optString("content")
                    if (content.isBlank()) {
                        throw IOException("Provider returned an empty response — try again, or use a different model/provider.")
                    }
                    content
                }
                main.post { done(result) }
            }
        })
    }

    /**
     * Robust reply parsing: models sometimes honor the ||| delimiter and sometimes
     * return newline-separated lines — handle both, strip list markers and quotes.
     */
    internal fun parseReplies(content: String): List<String> {
        val cleaned = content.trim()
        val rawParts = if (cleaned.contains("|||")) cleaned.split("|||") else cleaned.lines()
        return rawParts
            .map { it.trim().trim('"') }
            .map { it.replace(Regex("^\\s*(?:\\d+[.)]|[-•*])\\s*"), "") }
            .filter { it.isNotEmpty() }
            .take(3)
            .ifEmpty { throw IOException("Model returned no usable replies.") }
    }
}
