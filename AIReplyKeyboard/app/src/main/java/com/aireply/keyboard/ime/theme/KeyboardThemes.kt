package com.aireply.keyboard.ime.theme

/** A complete keyboard palette — inspired by flagship themes in modern AI keyboards. */
data class KeyboardTheme(
    val id: String,
    val label: String,
    val kbBg: Int,          // keyboard canvas
    val keyBg: Int,         // letter keys
    val functionalBg: Int,  // shift / backspace / toolbar keys
    val text: Int,          // primary key text
    val textDim: Int,       // secondary / function key text
    val accent: Int,        // enter, active shift, selected chip, ✦
    val onAccent: Int,      // text on accent
    val chipStroke: Int     // outline color for chips
)

object KeyboardThemes {

    val gboardDark = KeyboardTheme(
        id = "gboard_dark", label = "Gboard Dark",
        kbBg = 0xFF202124.toInt(), keyBg = 0xFF3C4043.toInt(),
        functionalBg = 0xFF2E3134.toInt(), text = 0xFFE8EAED.toInt(),
        textDim = 0xFF9AA0A6.toInt(), accent = 0xFF8AB4F8.toInt(),
        onAccent = 0xFF202124.toInt(), chipStroke = 0xFF5F6368.toInt()
    )

    val cyberpunkNeon = KeyboardTheme(
        id = "cyberpunk_neon", label = "Cyberpunk Neon",
        kbBg = 0xFF07060D.toInt(), keyBg = 0xFF1A162B.toInt(),
        functionalBg = 0xFF141122.toInt(), text = 0xFFF8FAFC.toInt(),
        textDim = 0xFF67E8F9.toInt(), accent = 0xFF00E5FF.toInt(),
        onAccent = 0xFF07060D.toInt(), chipStroke = 0xFF2C2448.toInt()
    )

    val midnightOled = KeyboardTheme(
        id = "midnight_oled", label = "Midnight OLED",
        kbBg = 0xFF000000.toInt(), keyBg = 0xFF121214.toInt(),
        functionalBg = 0xFF0A0A0C.toInt(), text = 0xFFFFFFFF.toInt(),
        textDim = 0xFF71717A.toInt(), accent = 0xFF38BDF8.toInt(),
        onAccent = 0xFF0A0A0C.toInt(), chipStroke = 0xFF27272A.toInt()
    )

    val neuralPurple = KeyboardTheme(
        id = "neural_purple", label = "Neural Purple",
        kbBg = 0xFF0D0A1A.toInt(), keyBg = 0xFF1A162B.toInt(),
        functionalBg = 0xFF141122.toInt(), text = 0xFFF8FAFC.toInt(),
        textDim = 0xFFA78BFA.toInt(), accent = 0xFFF43F5E.toInt(),
        onAccent = 0xFFFFFFFF.toInt(), chipStroke = 0xFF2C2448.toInt()
    )

    val auroraLight = KeyboardTheme(
        id = "aurora_light", label = "Aurora Light",
        kbBg = 0xFFE9EEF6.toInt(), keyBg = 0xFFFFFFFF.toInt(),
        functionalBg = 0xFFDDE3EC.toInt(), text = 0xFF1A1C1E.toInt(),
        textDim = 0xFF5F6368.toInt(), accent = 0xFF0B57D0.toInt(),
        onAccent = 0xFFFFFFFF.toInt(), chipStroke = 0xFFC4C7CC.toInt()
    )

    val all = listOf(gboardDark, cyberpunkNeon, midnightOled, neuralPurple, auroraLight)
}
