package com.aireply.keyboard.ai

/**
 * Game knowledge base, ported from the DMAssistant bot's KNOWLEDGE_BASE.
 * Powers the 💡 coach tip shown alongside generated replies.
 */
object AdviceKB {

    private val tips: Map<String, List<String>> = mapOf(
        "opener" to listOf(
            "Match her energy — short reply gets a short reply back",
            "Comment on something immediate: her last message, her story, her caption",
            "Specific beats generic: 'that chai reel? obsessed' > 'hey beautiful'",
            "Open-ended hooks only: questions that need more than yes/no",
            "Keep it under 20 words — punchy and direct",
            "Tease lightly, don't impress — curiosity beats try-hard",
            "Reference her world: her bio, her pet, her last post (Carnegie rule)",
            "Drop the period in texts — it reads less sincere (Klin, 2016)",
            "Desi opener: 'quick, chai or coffee — wrong answers get blocked' 😄",
            "India: opinion-bait wins — 'confirm or deny: momos are overrated'",
            "Never a bare 'hey' — Indian apps ignore generic openers"
        ),
        "high_enthu" to listOf(
            "She's excited — match the energy, add 1 emoji max",
            "Follow up on what SHE brought up, not a new topic",
            "Playful tease lands best when she's already laughing",
            "Brief self-reveal now: one small story earns her story back",
            "Deepen with 'why's that your favorite?' style hooks"
        ),
        "low_dry" to listOf(
            "Dry replies = low interest. One more value text, then stop chasing",
            "Abundance mindset: don't double-text the same day",
            "Shift topic lightly or exit gracefully: 'nice chatting — enjoy your day'",
            "Never send 'why no reply' — it reads as needy",
            "Wait 24h before trying again; send something worth answering",
            "Be responsive, not instant — no timed games (Spielmann, 2016)"
        ),
        "followup" to listOf(
            "Mirror her keyword: repeat her word and expand the question",
            "Facts → feelings: 'I like hiking' deserves 'what do you love about it?'",
            "One question per message — never stack three",
            "After 2-3 exchanges, move to a plan: specific day, specific place",
            "'Really? tell me more' works better than changing the subject",
            "Escalate slowly: facts → opinions → feelings (Aron's 36-questions method)",
            "Suggest an exciting date — arousal amplifies attraction (Dutton & Aron, 1974)"
        ),
        "mindset" to listOf(
            "Attraction is inversely proportional to neediness (Manson, 'Models')",
            "Honesty with zero expectation reads as pure confidence",
            "Concrete invites win: 'Saturday 7pm, that ramen place' > 'we should hang out'",
            "Leave the date at its peak — peak-end rule (Kahneman)",
            "Confirm the morning of: reliable, not needy",
            "Rejection is data — find YOUR pattern before blaming hers"
        ),
        "compliment" to listOf(
            "Praise taste, humor, energy — never looks or body early on",
            "Specific and genuine: 'your timing in that reel is hilarious'",
            "Hinglish twist lands well: 'ekdum jhakaas take yaar'",
            "Compliment + question: 'bold bio. what's the story behind it?'"
        ),
        "reel" to listOf(
            "Reels: play along — 'this slaps, what's the backstory? 👀'",
            "Duet/stitch energy: 'duet this if you're brave'",
            "Relatable + question: 'FYP knows me too well — your take?'",
            "Never just 'lol' — always add value or a question"
        ),
        "love_bombing" to listOf(
            "Too intense too fast = slow down. Ask about her day, hobbies",
            "Stay calm, don't match the intensity: 'that's sweet — tell me more about you'",
            "Pivot to light: 'haha manifesting slower — weekend plans?'"
        ),
        "money_talk" to listOf(
            "Money talk early = pivot to her interests, not your wallet",
            "Talk passions and values: 'cool — but what's your passion project?'",
            "If it persists, graceful exit beats negotiation"
        ),
        "rejection_signal" to listOf(
            "Uninterested cues: exit gracefully, don't push",
            "Accept declines warmly: 'cool, no worries — take care'",
            "Practice micro-convos daily; goal is authentic exchange, not force"
        )
    )

    /** One coaching tip for this moment, shown as 💡 next to the suggestions. */
    fun tip(mood: String, trap: String?): String {
        trap?.let { t -> tips[t]?.let { pool -> return pool.random() } }
        return when (mood) {
            "high_enthu" -> (tips["high_enthu"]!! + tips["followup"]!!).random()
            "low_dry" -> (tips["low_dry"]!! + tips["mindset"]!!).random()
            else -> (tips["followup"]!! + tips["compliment"]!! + tips["opener"]!! + tips["mindset"]!!).random()
        }
    }
}
