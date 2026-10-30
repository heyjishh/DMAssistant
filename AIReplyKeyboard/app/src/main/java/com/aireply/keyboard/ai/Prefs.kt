package com.aireply.keyboard.ai

import android.content.Context
import android.content.SharedPreferences
import org.json.JSONArray
import org.json.JSONObject

object Prefs {
    private const val FILE = "ai_reply_prefs"
    const val DEFAULT_BASE_URL = "https://api.openai.com/v1"
    const val DEFAULT_MODEL = "gpt-4o-mini"
    const val DEFAULT_VISION_MODEL = "gpt-4o-mini"
    private const val MAX_SAVED = 50

    private fun sp(context: Context): SharedPreferences =
        context.getSharedPreferences(FILE, Context.MODE_PRIVATE)

    fun baseUrl(context: Context): String =
        sp(context).getString("base_url", DEFAULT_BASE_URL)!!.trim().removeSuffix("/")

    fun apiKey(context: Context): String =
        sp(context).getString("api_key", "")!!.trim()

    fun model(context: Context): String =
        sp(context).getString("model", DEFAULT_MODEL)!!.trim()

    fun visionModel(context: Context): String =
        sp(context).getString("vision_model", DEFAULT_VISION_MODEL)!!.trim()

    /** Free-text description of the user's own texting style, injected into every prompt. */
    fun styleProfile(context: Context): String =
        sp(context).getString("style_profile", "")!!.trim()

    fun defaultTone(context: Context): Tone =
        runCatching { Tone.valueOf(sp(context).getString("default_tone", Tone.CASUAL.name)!!) }
            .getOrDefault(Tone.CASUAL)

    fun bilingual(context: Context): Boolean = sp(context).getBoolean("bilingual", false)

    /** Keyboard height scale: 0 = compact, 1 = normal, 2 = tall. */
    fun kbScaleIndex(context: Context): Int = sp(context).getInt("kb_scale_idx", 1)

    fun kbScale(context: Context): Float = when (kbScaleIndex(context)) {
        0 -> 0.85f
        2 -> 1.2f
        else -> 1.0f
    }

    fun setKbScaleIndex(context: Context, index: Int) {
        sp(context).edit().putInt("kb_scale_idx", index).apply()
    }

    /** Remembers the last tone used per chat app, so each conversation keeps its vibe. */
    fun toneForApp(context: Context, packageName: String): Tone? {
        val raw = sp(context).getString("tone_map", "{}")!!
        return runCatching {
            JSONObject(raw).optString(packageName).takeIf { it.isNotEmpty() }?.let { Tone.valueOf(it) }
        }.getOrNull()
    }

    fun setToneForApp(context: Context, packageName: String, tone: Tone) {
        val map = runCatching { JSONObject(sp(context).getString("tone_map", "{}")!!) }
            .getOrDefault(JSONObject())
        map.put(packageName, tone.name)
        sp(context).edit().putString("tone_map", map.toString()).apply()
    }

    /** Packages whose screen content must never be captured. */
    fun excludedApps(context: Context): List<String> =
        sp(context).getString("excluded_apps", "")!!
            .split(",").map { it.trim() }.filter { it.isNotEmpty() }

    fun save(context: Context, baseUrl: String, apiKey: String, model: String, tone: Tone) {
        sp(context).edit()
            .putString("base_url", baseUrl.trim().ifEmpty { DEFAULT_BASE_URL })
            .putString("api_key", apiKey.trim())
            .putString("model", model.trim().ifEmpty { DEFAULT_MODEL })
            .putString("default_tone", tone.name)
            .apply()
    }

    fun saveAdvanced(context: Context, visionModel: String, styleProfile: String) {
        sp(context).edit()
            .putString("vision_model", visionModel.trim().ifEmpty { DEFAULT_VISION_MODEL })
            .putString("style_profile", styleProfile.trim())
            .apply()
    }

    fun saveExtras(context: Context, bilingual: Boolean, excludedApps: String) {
        sp(context).edit()
            .putBoolean("bilingual", bilingual)
            .putString("excluded_apps", excludedApps.trim())
            .apply()
    }

    // --- Saved replies ---

    fun savedReplies(context: Context): List<String> {
        val raw = sp(context).getString("saved_replies", "[]")!!
        return runCatching {
            val arr = JSONArray(raw)
            (0 until arr.length()).map { arr.getString(it) }
        }.getOrDefault(emptyList())
    }

    fun addSaved(context: Context, text: String) {
        val current = savedReplies(context).toMutableList()
        if (current.contains(text)) return
        current.add(0, text)
        while (current.size > MAX_SAVED) current.removeAt(current.size - 1)
        sp(context).edit().putString("saved_replies", JSONArray(current).toString()).apply()
    }

    fun removeSaved(context: Context, index: Int) {
        val current = savedReplies(context).toMutableList()
        if (index in current.indices) {
            current.removeAt(index)
            sp(context).edit().putString("saved_replies", JSONArray(current).toString()).apply()
        }
    }
}
