package com.aireply.keyboard.ai

/**
 * Research- and book-backed game playbook. Sources include:
 * - Dutton & Aron (1974) misattribution of arousal; Aron et al. (1997) 36 questions
 * - Montoya & Horton (2012) reciprocity of liking; Klin (2016) punctuation in texting
 * - Spielmann et al. (2016) responsiveness; UT Austin emoji research
 * - Mark Manson "Models" (non-neediness); Levine & Heller "Attached" (attachment styles)
 * - Dale Carnegie "How to Win Friends"; Kahneman peak-end rule
 */
object Playbook {

    data class Entry(val title: String, val action: String, val why: String)

    val sections: List<Pair<String, List<Entry>>> = listOf(
        "Openers & First Contact" to listOf(
            Entry(
                "Reference her world",
                "Open with something specific from her profile, story, or last message — a place, a pet, a line from her bio.",
                "Carnegie's #1 rule: people light up when you talk about their interests. Specific openers outperform 'hey' every time."
            ),
            Entry(
                "The 36-questions principle",
                "Escalate gradually: facts → opinions → feelings. Match her depth; go one notch deeper than she does.",
                "Aron et al. (1997) built closeness between strangers with graduated, reciprocal self-disclosure."
            ),
            Entry(
                "Show genuine interest early",
                "Say you like something about her — her humor, her taste — plainly and once.",
                "Reciprocity of liking is one of the most robust findings in relationship science (Montoya, 2012): knowing you're liked makes you more likeable."
            ),
            Entry(
                "Effort matches effort",
                "Match her message length and energy, then add one small thing of your own.",
                "Mirror-and-add keeps conversations balanced instead of one-sided interviews."
            )
        ),
        "Texting Science" to listOf(
            Entry(
                "Drop the period",
                "End texts without a full stop ('sounds good' not 'sounds good.').",
                "Klin (2016): periods in texts are perceived as less sincere — punctuation reads as cold or formal."
            ),
            Entry(
                "One or two emojis, matched",
                "Mirror her emoji style: if she uses 😂, you can too; if none, stay clean.",
                "UT Austin research: emojis increase perceived warmth and closeness — but mismatched emoji hurt clarity."
            ),
            Entry(
                "Be responsive, not instant",
                "Reply within 10-15 minutes when free. No timed games.",
                "Spielmann et al. (2016): responsiveness strongly signals interest and is rated more attractive than playing hard to get."
            ),
            Entry(
                "Voice note, 15 seconds",
                "Send a short voice note with energy and a smile in your voice.",
                "Voice carries warmth and humor that text flattens — it differentiates you instantly."
            ),
            Entry(
                "Funny first, give it room",
                "If your opener is funny, let it land. Don't double-text to explain the joke.",
                "Humor is one of the strongest attraction predictors (Greengross, 2011) — jokes need silence to work."
            )
        ),
        "Conversation Depth" to listOf(
            Entry(
                "Facts → feelings",
                "When she shares a fact ('I love hiking'), ask about the feeling ('what do you love about it?').",
                "This is the Aron disclosure engine — emotional questions build intimacy fast."
            ),
            Entry(
                "Misattribution of arousal",
                "Suggest dates with excitement built in: amusement park, spicy food, dance class, scary movie.",
                "Dutton & Aron (1974): arousal from any source gets partly attributed to attraction to you."
            ),
            Entry(
                "Shared laughter beats impressing",
                "Make fun observations about shared situations; laugh at her jokes genuinely.",
                "Humor production AND appreciation both predict attraction (Greengross & Miller, 2011)."
            ),
            Entry(
                "Novelty bonds",
                "New experiences together — new café, new activity — bond faster than routine.",
                "Self-expansion theory (Aron): doing novel things with someone creates relationship-like closeness."
            ),
            Entry(
                "Remember the details",
                "Bring back something she said days ago: 'how did that exam go?'",
                "Dale Carnegie: remembered details are the strongest signal of genuine interest."
            )
        ),
        "Emotional Game" to listOf(
            Entry(
                "Non-neediness is the only skill",
                "Have your own plans and opinions; never negotiate for her attention.",
                "Mark Manson, 'Models': attraction is inversely proportional to neediness."
            ),
            Entry(
                "Honesty as a gift",
                "Say what you actually think and want, with zero expectation attached.",
                "'Models': unconditional honesty reads as confidence and separates you from players."
            ),
            Entry(
                "Spot attachment styles",
                "Secure people text consistently. Anxious escalate fast. Avoidant pull away when close.",
                "Levine & Heller, 'Attached': matching with secure partners avoids the anxious-avoidant trap."
            ),
            Entry(
                "Uncertainty — use once",
                "One slightly unpredictable reply spikes her attention. Once. Then be warm.",
                "Whitchurch, Wilson & Gilbert: uncertainty increases thinking about someone — but as a strategy it backfires."
            ),
            Entry(
                "Let her invest",
                "Ask small favors and let her plan part of the date.",
                "Ben Franklin effect: we like people we invest in, not just people who invest in us."
            )
        ),
        "Dates & Escalation" to listOf(
            Entry(
                "Concrete invites only",
                "'Saturday 7pm, that ramen place near Hauz Khas' — never 'we should hang out sometime'.",
                "Specific invites are easy to say yes to; vague ones die in 'maybe'."
            ),
            Entry(
                "Activity over interview",
                "First dates: walk, arcade, market, cooking class — not face-to-face dinner interrogation.",
                "Combines arousal research with self-expansion: activity + novelty + eye contact breaks."
            ),
            Entry(
                "Confirm the morning of",
                "One short message the morning of the date: 'still on for tonight 😄'.",
                "Responsiveness research: it raises show rates and reads as reliable, not needy."
            ),
            Entry(
                "End on the peak",
                "Leave while the conversation is at its best — not when it dies.",
                "Peak-end rule (Kahneman): memories are judged by the peak and the ending."
            )
        ),
        "India Game 🇮🇳" to listOf(
            Entry(
                "Personalize or get ignored",
                "Open with a specific detail from her bio or photos — never a bare 'hey' or 'you look beautiful'.",
                "Indian women on dating subs consistently say generic openers get ignored; specificity is the filter."
            ),
            Entry(
                "Kill the interview questions",
                "Avoid 'favorite books/movies?' — the most overused opener on Indian apps. Use opinion-bait instead.",
                "Indian dating subs call it tired; 'confirm or deny' and 'unpopular opinion' style hooks get replies."
            ),
            Entry(
                "Food is the universal hook",
                "Chai vs coffee, momos takes, best street food in her city — playful food debates are low-risk and endless.",
                "Chai + food culture is a natural shared topic; food debates give her an easy, fun way to reply."
            ),
            Entry(
                "Cricket and IPL banter",
                "Light team rivalry or match reactions — 'your team lost yesterday, need space?'",
                "Cricket is shared cultural ground; banter about it reads playful, not creepy."
            ),
            Entry(
                "Festivals are free re-entry hooks",
                "Diwali, Holi, New Year — a warm festival text restarts dead chats naturally.",
                "Culturally expected and welcome; low-pressure way to reconnect after silence."
            ),
            Entry(
                "Don't rush Instagram/number",
                "Earn the handle: build rapport first. Asking in the first 3 messages kills trust.",
                "Indian dating-safety reports show harassment is the #1 complaint — patience reads as safe, and safe gets replies."
            ),
            Entry(
                "She may not text first — that's cultural",
                "If she matched but doesn't initiate, carry the conversation warmly. Silence isn't rejection.",
                "Indian women often hesitate to initiate due to cultural norms — low-effort from her side ≠ disinterest."
            ),
            Entry(
                "Shaadi humor, lightly",
                "Playful jabs at wedding season and 'log kya kahenge' land well AFTER rapport — never as the opener.",
                "Shared desi experience = instant inside joke, but only once comfort exists."
            ),
            Entry(
                "Safety first, always",
                "No sexual comments early, no pressure to meet or share info. Respect is the filter she's screening with.",
                "Studies (Rajan, 2025) show Indian women's top app concern is harassment — respectful guys stand out immediately."
            )
        ),
        "Mindset" to listOf(
            Entry(
                "Abundance is action",
                "Talk to people daily. Options are built, not found.",
                "Scarcity mindset creates neediness — the #1 attraction killer (Manson, 'Models')."
            ),
            Entry(
                "Detach from the outcome",
                "Invest in your own judgment and standards, not in her approval of you.",
                "'Models': the people you date reflect your emotional state; neediness is valuing her view over yours."
            ),
            Entry(
                "Rejection is data",
                "Every dead chat shows you a pattern. Look for yours before blaming theirs.",
                "'Models': pattern recognition is how you actually improve — reaction reveals beliefs."
            )
        )
    )

    /** Flattened into library-style categories (title + action + why per item). */
    fun asCategories(): List<LibraryData.Category> = sections.map { (section, entries) ->
        LibraryData.Category(
            "📚 $section",
            entries.map { "${it.title}\n${it.action}\nWhy it works: ${it.why}" }
        )
    }
}
