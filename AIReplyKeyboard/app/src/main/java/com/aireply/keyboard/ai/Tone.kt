package com.aireply.keyboard.ai

enum class Tone(val label: String, val instruction: String) {
    CASUAL(
        "Casual",
        "relaxed, friendly and natural — like texting a friend, everyday language, light contractions"
    ),
    FLIRTY(
        "Flirty",
        "playful, charming and flirty — natural compliments, light teasing, keeps it tasteful and fun"
    ),
    PROFESSIONAL(
        "Professional",
        "polite, clear and professional — courteous, well-structured, no slang"
    ),
    FUNNY(
        "Funny",
        "witty and humorous — clever jokes or playful exaggeration, never mean"
    ),
    DIRECT(
        "Direct",
        "short, confident and straight to the point — no filler"
    ),
    SPICY(
        "Spicy",
        "bold and suggestive adult flirting — escalating tension and innuendo, still tasteful rather than graphic"
    ),
    CONFIDENT(
        "Confident",
        "self-assured and bold — easy charisma, no neediness, owns the conversation"
    ),
    SAVAGE(
        "Savage",
        "sharp, witty teasing — playful roasts and bold comebacks, edgy but never cruel or about appearance"
    ),
    SMOOTH(
        "Smooth",
        "effortlessly charming — calm charisma, clever low-key compliments, unhurried"
    );

    companion object {
        fun fromLabel(label: String): Tone = entries.first { it.label == label }
    }
}
