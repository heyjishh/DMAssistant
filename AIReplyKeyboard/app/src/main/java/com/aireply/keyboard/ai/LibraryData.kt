package com.aireply.keyboard.ai

/** Offline message library — categorized lines users can copy without calling the API. */
object LibraryData {

    data class Category(val label: String, val items: List<String>)

    val categories: List<Category> = listOf(
        Category(
            "Pickup Lines",
            listOf(
                "Are you a parking ticket? Because you've got fine written all over you.",
                "Do you have a name, or can I call you mine?",
                "I was going to say something really smooth, but I forgot it the second I saw you.",
                "On a scale of 1 to America, how free are you tonight?",
                "Do you believe in love at first text, or should I message you again?",
                "I'm not a photographer, but I can picture us together.",
                "If you were a vegetable, you'd be a cute-cumber.",
                "Are you Wi-Fi? Because I'm feeling a connection.",
                "My phone's battery lasts longer than my resolve to not text you twice in a row.",
                "I'd say God bless you, but it looks like he already did.",
                "Is your dad a boxer? Because you're a knockout!",
                "I must be a snowflake, because I've fallen for you."
            )
        ),
        Category(
            "Icebreakers",
            listOf(
                "Two truths and a lie — you start, I'll try to guess.",
                "Quick poll: pineapple on pizza — crime or art?",
                "What's the most overrated thing everyone seems to love?",
                "You get one free trip anywhere, leaving tomorrow. Where are we going?",
                "What's a hill you'd die on that you think is totally reasonable?",
                "Coffee person or tea person? This decides everything, obviously.",
                "What's the best thing that happened to you this week?",
                "If your life had a theme song right now, what would it be?",
                "What's something you were obsessed with as a kid?",
                "Would you rather explore the ocean or space? And why is space the wrong answer?",
                "What app do you open first every morning? Be honest.",
                "Sell me on your favorite weird snack."
            )
        ),
        Category(
            "Compliments",
            listOf(
                "Your energy is genuinely contagious.",
                "You have great taste — I can tell from your playlist.",
                "Talking to you makes my day noticeably better.",
                "You're the kind of person people remember.",
                "Your laugh sounds like it fixes bad days.",
                "You make ordinary conversations interesting.",
                "I like how passionate you get about the things you love.",
                "You have one of those smiles that should be studied.",
                "You're refreshingly easy to talk to.",
                "Whatever you're doing with your life, it's working.",
                "You have main character energy, and I mean that in the best way.",
                "The world needs more people like you — glad I found one."
            )
        ),
        Category(
            "Romantic Quotes",
            listOf(
                "In all the world, there is no heart for me like yours. — Maya Angelou",
                "I have loved the stars too fondly to be fearful of the night. — Sarah Williams",
                "You are my sun, my moon, and all my stars. — E.E. Cummings",
                "Whatever our souls are made of, his and mine are the same. — Emily Brontë",
                "To love and be loved is to feel the sun from both sides. — David Viscott",
                "I love you begins by I, but it ends up by you. — Charles de Leusse",
                "If I know what love is, it is because of you. — Hermann Hesse",
                "You don't find love, it finds you. It's got a little bit to do with destiny. — Anaïs Nin",
                "The best thing to hold onto in life is each other. — Audrey Hepburn",
                "I like you very much. Just as you are. — Jane Austen",
                "Love is composed of a single soul inhabiting two bodies. — Aristotle",
                "Grow old along with me! The best is yet to be. — Robert Browning"
            )
        ),
        Category(
            "POV Openers",
            listOf(
                "POV: My camera roll is 80% memes, 20% panic. Yours?",
                "POV: '5 more mins' turns into 3 hours. Same gang?",
                "POV: We ghost plans but overthink texts. Hi?",
                "POV: Chai fixes 90% of our problems. The other 10%?",
                "POV: We say 'I'm fine' but mean 'send memes'. You?",
                "POV: We play '2 truths 1 lie'. You start or I do?",
                "POV: Would you rather: no WiFi or no chai? Fight me",
                "POV: Rate my vibe 1-10. I'll rate yours back. Deal?",
                "POV: Send me a song. I'll send one back. No skips",
                "POV: Your bio is bold. I need the director's cut",
                "POV: We both gatekeep the best spots. Share one?",
                "POV: Your posts are my comfort show. What's the pilot episode?"
            )
        ),
        Category(
            "Reel Comments",
            listOf(
                "bro this is me at 3am",
                "the accuracy is scary",
                "send this to your group chat",
                "i felt this in my soul",
                "deadass relatable",
                "finally someone said it",
                "the algorithm knows me too well",
                "iykyk vibes",
                "duet this if you're brave",
                "reply with your version",
                "stitch challenge accepted?",
                "this reel just ended my scroll — what's the backstory? 👀"
            )
        ),
        Category(
            "Voice Note Scripts",
            listOf(
                "Yo! Your bio — how do you even live by that? I tried and lasted 3 days. Voice note me the real story.",
                "That content of yours? Obsessed. What's the one thing no one gets? Voice note me your hot take.",
                "Voice note tag: I say one word, you reply the first thing that comes to mind. Go.",
                "2 truths 1 lie: I cried in a movie, I can cook, I have 12 plants. Guess — voice note me!",
                "Okay that post stopped my scroll. Quick voice note: fave part and why?"
            )
        ),
        Category(
            "Emoji Openers",
            listOf(
                "chai question coffee answer ☕❓",
                "question meme answer 😂❓",
                "question song answer 🎵❓",
                "question fail answer 😩❓",
                "question and chill? ❄️❓"
            )
        ),
        Category(
            "No-Reply Nudges",
            listOf(
                "Seen but no reply? Fair. Now I'm IYKYK — your turn.",
                "I'll take the silence as you picking your fave song for us. Good choice.",
                "No pressure — whenever you're free, I want your take on this.",
                "Not double texting, just leaving the door open. 🚪",
                "Abundance mindset says move on… but one last question first 👀"
            )
        )
    )
}
