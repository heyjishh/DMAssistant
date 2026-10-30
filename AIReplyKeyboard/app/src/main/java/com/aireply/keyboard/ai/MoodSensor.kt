package com.aireply.keyboard.ai

/**
 * Fast keyword-based mood & "trap" detection, ported from the DMAssistant bot
 * (with word-boundary matching to avoid substring false positives like
 * "what are you up to" reading as defensive).
 */
object MoodSensor {

    data class Read(val mood: String, val trap: String?, val trapAdvice: String?)

    private fun hasWord(text: String, word: String): Boolean =
        Regex("\\b${Regex.escape(word)}\\b", RegexOption.IGNORE_CASE).containsMatchIn(text)

    private fun hasAny(text: String, words: List<String>): Boolean = words.any { hasWord(text, it) }

    fun read(conversation: String): Read {
        val lines = conversation.lines().filter { it.isNotBlank() }
        val lastLine = lines.lastOrNull()?.lowercase().orEmpty()
        val combined = lines.takeLast(3).joinToString(" ").lowercase()

        val mood = when {
            hasAny(combined, listOf("haha", "lol", "lmao", "omg", "yess", "slay", "deadass")) ||
                listOf("😂", "😍", "😊").any { it in combined } -> "high_enthu"
            lastLine.length < 25 && hasAny(lastLine, listOf("k", "ok", "kk", "hmm", "idk", "whatever", "busy", "nm")) -> "low_dry"
            hasAny(
                combined,
                listOf("wtf", "seriously", "back off", "not interested", "who is this", "stop texting", "stop messaging")
            ) -> "angry_defensive"
            else -> "medium_cool"
        }

        var trap: String? = null
        var advice: String? = null
        if (lines.size < 10 && (
                    hasAny(combined, listOf("soulmate", "married")) ||
                    hasAny(combined, listOf("love you", "need you", "you're perfect", "you are perfect"))
                    )
        ) {
            trap = "love_bombing"
            advice = "Too intense too fast — slow down, ask about their day instead of matching the intensity."
        } else if (hasAny(
                combined,
                listOf("expensive", "gift", "rich", "afford", "buy me", "pay for", "dinner on you")
            )
        ) {
            trap = "money_talk"
            advice = "Money-focused early — pivot to their interests and values, not your wallet."
        } else if (hasAny(
                combined,
                listOf("creep", "weirdo", "block", "leave me alone", "don't text me", "dont text me")
            )
        ) {
            trap = "rejection_signal"
            advice = "Uninterested cues — exit gracefully, don't push. Respect the boundary."
        }

        return Read(mood, trap, advice)
    }

    fun moodAdvice(mood: String): String = when (mood) {
        "high_enthu" -> "They're excited — match the energy, ask open questions to keep it flowing."
        "low_dry" -> "Short replies mean low energy — one more value-adding text max, then pause. Don't chase."
        "angry_defensive" -> "They're defensive — stay calm, don't match the intensity, or exit gracefully."
        else -> "Neutral mood — keep it light and curious."
    }
}
