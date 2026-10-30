package com.aireply.keyboard.ai

/**
 * Offline fallback suggestions when the API is unreachable or no key is configured.
 * Mood-aware canned replies in the spirit of the DMAssistant bot's quota fallbacks —
 * the ✦ button never dead-ends.
 */
object OfflineFallback {

    fun suggest(tone: Tone, mood: String, hinglish: Boolean): List<String> {
        val pool = if (hinglish) hinglishPool(tone, mood) else englishPool(tone, mood)
        return pool.take(3)
    }

    private fun englishPool(tone: Tone, mood: String): List<String> {
        val base = when (mood) {
            "high_enthu" -> listOf(
                "Okay I'm invested now — tell me the whole story",
                "Ha, you're dangerous. What else you got?",
                "This energy is great. What's the story behind it?"
            )
            "low_dry" -> listOf(
                "Fair enough — I'll let you breathe. Talk later?",
                "Cool, no rush. Something made me think of you today though",
                "Alright, keeping it light — how's your week been?"
            )
            "angry_defensive" -> listOf(
                "Didn't mean to push — my bad. Tell me what's up",
                "Fair. I'll back off — I'm around if you want to talk",
                "You're right to feel that way. What can I do better?"
            )
            else -> when (tone) {
                Tone.PROFESSIONAL -> listOf(
                    "Thanks for the update — happy to help if anything else comes up.",
                    "Understood. Shall we plan the next step?",
                    "Noted — I'll follow up with the details shortly."
                )
                Tone.FUNNY, Tone.SAVAGE -> listOf(
                    "I was going to say something clever but you've clearly already won today",
                    "On a scale of 1 to 10, how tired are you of my messages? Be honest",
                    "Quick poll: should I keep texting or pretend I have a life?"
                )
                else -> listOf(
                    "Okay now I'm curious — what's the story there?",
                    "Ha, good one. What have you been up to?",
                    "That actually made my day. How's yours going?"
                )
            }
        }
        return base
    }

    private fun hinglishPool(tone: Tone, mood: String): List<String> {
        return when (mood) {
            "high_enthu" -> listOf(
                "Arre ab toh full story chahiye 👀",
                "Haha tu dangerous hai yaar. Aur bata",
                "Yeh energy mast hai. Kya scene tha usme?"
            )
            "low_dry" -> listOf(
                "Theek hai yaar, koi na. Baad mein baat karte hain",
                "Cool, no rush. Aaj tujhe yaad kiya btw",
                "Chal light rakhte hain — week kaisa ja raha hai?"
            )
            "angry_defensive" -> listOf(
                "Arre push nahi karna tha, sorry. Bata kya hua",
                "Fair hai. Main back off karta hoon — baat karni ho toh main hoon",
                "Sahi feel kar rhi hai tu. Main kya better kar sakta hoon?"
            )
            else -> listOf(
                "Ab toh curious ho gaya — poori story suna",
                "Haha ekdum sahi. Kya kar rhi hai aajkal?",
                "Wah, din ban gaya. Tera kaisa chal raha hai?"
            )
        }
    }
}
