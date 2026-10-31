package com.aireply.keyboard.ime

import android.graphics.drawable.ColorDrawable
import android.inputmethodservice.InputMethodService
import android.text.TextUtils
import android.view.Gravity
import android.view.KeyEvent
import android.view.View
import android.widget.PopupWindow
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import com.aireply.keyboard.ai.AdviceKB
import com.aireply.keyboard.ai.LlmClient
import com.aireply.keyboard.ai.MoodSensor
import com.aireply.keyboard.ai.OfflineFallback
import com.aireply.keyboard.ai.Prefs
import com.aireply.keyboard.ai.Tone
import com.aireply.keyboard.reader.ChatStore

class AiKeyboardService : InputMethodService() {

    private lateinit var root: LinearLayout
    private lateinit var keysColumn: LinearLayout
    private lateinit var toneBtn: Button
    private lateinit var generateBtn: Button
    private lateinit var suggestionButtons: List<Button>
    private lateinit var tipView: TextView

    private var shiftOn = false
    private var symbolsOn = false
    private var symbolPage = 0
    private var emojiOn = false
    private var tone: Tone = Tone.CASUAL
    private var popup: PopupWindow? = null

    private var lastSnapshot: String = ""
    private var lastReplies: List<String> = emptyList()
    private var builtConfig = -1 // theme idx * 10 + kb scale idx, to detect settings changes

    // Live "Generating… Ns" timer while waiting on the AI call.
    private val timerHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private var genStart = 0L

    private fun startGenTimer() {
        if (!::generateBtn.isInitialized) return
        genStart = System.currentTimeMillis()
        val tick = object : Runnable {
            override fun run() {
                if (::generateBtn.isInitialized && generateBtn.visibility == View.VISIBLE) {
                    val s = (System.currentTimeMillis() - genStart) / 1000.0
                    generateBtn.text = String.format("✦  Generating… %.0fs", s)
                    timerHandler.postDelayed(this, 500)
                }
            }
        }
        timerHandler.post(tick)
    }

    private fun genSeconds(): String = "%.1fs".format((System.currentTimeMillis() - genStart) / 1000.0)

    private fun stopGenTimer() {
        timerHandler.removeCallbacksAndMessages(null)
    }

    private val emojiRows = listOf(
        listOf("😀", "😂", "🥲", "😍", "😘", "😜", "🤔", "😏", "😎", "🥳", "😇"),
        listOf("😢", "😭", "😡", "🙄", "😴", "🤗", "🤩", "🫣", "🤠", "😈", "👀"),
        listOf("👍", "👏", "🙌", "👌", "✌️", "🤞", "🫶", "❤️", "💔", "🔥", "💯"),
        listOf("✨", "🎉", "🌹", "☕", "🍕", "🎮", "⚽", "🎧", "📸", "💀", "🤡")
    )

    // ── Theme (applied from KeyboardThemes in onCreate) ─────────────────────
    private var KB_BG = 0xFF202124.toInt()
    private var KEY_BG = 0xFF3C4043.toInt()
    private var SPECIAL_BG = 0xFF2E3134.toInt()
    private var CHIP_STROKE = 0xFF5F6368.toInt()
    private var TEXT = 0xFFE8EAED.toInt()
    private var TEXT_DIM = 0xFF9AA0A6.toInt()
    private var ACCENT_BLUE = 0xFF8AB4F8.toInt()
    private var ON_ACCENT = 0xFF202124.toInt()

    private fun applyTheme() {
        val t = com.aireply.keyboard.ime.theme.KeyboardThemes.all[
            Prefs.themeIndex(this).coerceIn(0, com.aireply.keyboard.ime.theme.KeyboardThemes.all.lastIndex)
        ]
        KB_BG = t.kbBg
        KEY_BG = t.keyBg
        SPECIAL_BG = t.functionalBg
        TEXT = t.text
        TEXT_DIM = t.textDim
        ACCENT_BLUE = t.accent
        ON_ACCENT = t.onAccent
        CHIP_STROKE = t.chipStroke
    }

    private fun dp(v: Float): Float = v * resources.displayMetrics.density

    /** Rounded key background with press ripple, Gboard-style. */
    private fun keyBackground(fill: Int): android.graphics.drawable.Drawable {
        val shape = android.graphics.drawable.GradientDrawable().apply {
            setColor(fill)
            cornerRadius = dp(12f)
        }
        val mask = android.graphics.drawable.GradientDrawable().apply {
            setColor(0xFFFFFFFF.toInt())
            cornerRadius = dp(12f)
        }
        return android.graphics.drawable.RippleDrawable(
            android.content.res.ColorStateList.valueOf(0x30FFFFFF), shape, mask
        )
    }

    private fun outlineBackground(): android.graphics.drawable.Drawable {
        val shape = android.graphics.drawable.GradientDrawable().apply {
            setColor(KB_BG)
            cornerRadius = dp(22f)
            setStroke(dp(1f).toInt(), CHIP_STROKE)
        }
        val mask = android.graphics.drawable.GradientDrawable().apply {
            setColor(0xFFFFFFFF.toInt())
            cornerRadius = dp(22f)
        }
        return android.graphics.drawable.RippleDrawable(
            android.content.res.ColorStateList.valueOf(0x28FFFFFF), shape, mask
        )
    }

    private val accentRows = listOf(
        listOf('q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'),
        listOf('a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'),
        listOf('z', 'x', 'c', 'v', 'b', 'n', 'm')
    )
    // Gboard-style symbol pages: page 0 = numbers/common, page 1 = brackets/currency.
    private val symbolPages = listOf(
        listOf(
            listOf('1', '2', '3', '4', '5', '6', '7', '8', '9', '0'),
            listOf('@', '#', '$', '%', '&', '-', '+', '(', ')'),
            listOf('=', '*', '"', '\'', ':', ';', '!')
        ),
        listOf(
            listOf('[', ']', '{', '}', '<', '>', '~', '`', '^'),
            listOf('_', '\\', '|', '€', '£', '¥', '₹', '•', '°'),
            listOf('+', '=', '~', '%', '^', '&', '*')
        )
    )

    companion object {
        @JvmStatic
        @Volatile
        var instance: AiKeyboardService? = null
            private set

        private val ACCENTS: Map<Char, List<Char>> = mapOf(
            'a' to listOf('á', 'à', 'â', 'ã', 'ä'),
            'e' to listOf('é', 'è', 'ê', 'ë'),
            'i' to listOf('í', 'ì', 'î', 'ï'),
            'o' to listOf('ó', 'ò', 'ô', 'õ', 'ö'),
            'u' to listOf('ú', 'ù', 'û', 'ü'),
            'n' to listOf('ñ'),
            'c' to listOf('ç'),
            's' to listOf('ß'),
            'y' to listOf('ý', 'ÿ')
        )
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
        applyTheme()
        tone = Prefs.defaultTone(this)
    }

    override fun onDestroy() {
        if (instance === this) instance = null
        dismissPopup()
        super.onDestroy()
    }

    /** Debug test hooks — driven by TestHookReceiver in debug builds only. */
    fun debugGenerate() = generateReplies()

    fun debugCapture() = startLiveCapture()

    fun rebuildWithTheme() {
        applyTheme()
        if (::suggestionButtons.isInitialized) setInputView(onCreateInputView())
    }

    fun debugTone(name: String) {
        tone = runCatching { Tone.valueOf(name.uppercase()) }.getOrDefault(tone)
        updateToneChip()
    }

    fun debugInsert(index: Int) {
        val text = lastReplies.getOrElse(index) { return }
        insertSuggestion(text, index)
    }

    override fun onCreateInputView(): View {
        val density = resources.displayMetrics.density
        val px: (Int) -> Int = { dp -> (dp * density).toInt() }

        root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(KB_BG)
            setPadding(px(4), px(6), px(4), px(8))
        }
        builtConfig = Prefs.themeIndex(this) * 10 + Prefs.kbScaleIndex(this)

        // --- AI controls row: tone selector + capture / rewrite / keyboard switch ---
        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        toneBtn = Button(this).apply {
            text = "🎭 ${tone.label} ▾"
            textSize = 13f
            isAllCaps = false
            setSingleLine()
            background = keyBackground(SPECIAL_BG)
            setTextColor(TEXT)
            layoutParams = LinearLayout.LayoutParams(0, px(40), 1f).apply { setMargins(px(3), 0, px(3), 0) }
            setOnClickListener { v ->
                v.performHapticFeedback(android.view.HapticFeedbackConstants.KEYBOARD_TAP)
                showTonePicker(v, px)
            }
        }
        controls.addView(toneBtn)
        val tools = listOf(
            "📸" to { startLiveCapture() },
            "✎" to { rewriteDraft() }
        )
        tools.forEach { (icon, action) ->
            controls.addView(Button(this).apply {
                text = icon
                textSize = 16f
                background = keyBackground(SPECIAL_BG)
                setTextColor(TEXT_DIM)
                layoutParams = LinearLayout.LayoutParams(px(46), px(40)).apply { setMargins(px(3), 0, 0, 0) }
                setOnClickListener { v ->
                    v.performHapticFeedback(android.view.HapticFeedbackConstants.KEYBOARD_TAP)
                    action()
                }
            })
        }
        root.addView(controls, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { topMargin = px(4) })

        // --- Replies panel: big generate button, then stacked full-width reply cards ---
        val panel = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        generateBtn = Button(this).apply {
            text = "✦  Generate AI replies"
            textSize = 15f
            isAllCaps = false
            background = keyBackground(ACCENT_BLUE)
            setTextColor(ON_ACCENT)
            stateListAnimator = null
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, px(48)
            ).apply { setMargins(px(3), px(4), px(3), 0) }
            setOnClickListener { v ->
                v.performHapticFeedback(android.view.HapticFeedbackConstants.KEYBOARD_TAP)
                generateReplies()
            }
        }
        panel.addView(generateBtn)
        suggestionButtons = (0 until 3).map { index ->
            Button(this).apply {
                textSize = 14f
                isAllCaps = false
                background = keyBackground(KEY_BG)
                setTextColor(TEXT)
                gravity = Gravity.START or Gravity.CENTER_VERTICAL
                maxLines = 2
                ellipsize = TextUtils.TruncateAt.END
                setPadding(px(12), px(6), px(12), px(6))
                visibility = View.GONE
                stateListAnimator = null
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { setMargins(px(3), px(3), px(3), 0); minHeight = px(46) }
                setOnClickListener { insertSuggestion(text.toString(), index) }
                setOnLongClickListener {
                    val t = text.toString()
                    if (t.isBlank()) false
                    else {
                        Prefs.addSaved(this@AiKeyboardService, t)
                        Toast.makeText(this@AiKeyboardService, "Saved 🔖 — find it in the app", Toast.LENGTH_SHORT).show()
                        true
                    }
                }
            }.also { panel.addView(it) }
        }

        // --- 💡 Coach tip line (from the game knowledge base) ---
        tipView = TextView(this).apply {
            setTextColor(TEXT_DIM)
            textSize = 11f
            maxLines = 2
            ellipsize = TextUtils.TruncateAt.END
            setPadding(px(12), px(2), px(12), px(2))
            visibility = View.GONE
        }
        panel.addView(tipView, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { topMargin = px(2) })

        resetSuggestions()
        root.addView(panel, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { topMargin = px(2) })

        // --- QWERTY keys ---
        keysColumn = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        root.addView(keysColumn, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { topMargin = px(6) })

        rebuildKeys(px)
        return root
    }

    /** Tone selector chip label/style. */
    private fun updateToneChip() {
        if (!::toneBtn.isInitialized) return
        toneBtn.text = "🎭 ${tone.label} ▾"
        toneBtn.background = keyBackground(ACCENT_BLUE)
        toneBtn.setTextColor(ON_ACCENT)
    }

    /** Grid popup with all 9 tones — replaces the cramped chip row. */
    private fun showTonePicker(anchor: View, px: (Int) -> Int) {
        dismissPopup()
        val scrim = (0xF2 shl 24) or (KB_BG and 0x00FFFFFF)
        val grid = android.widget.GridLayout(this).apply {
            rowCount = 3
            columnCount = 3
            background = ColorDrawable(scrim)
            setPadding(px(8), px(8), px(8), px(8))
        }
        Tone.entries.forEach { t ->
            grid.addView(Button(this).apply {
                text = t.label
                textSize = 13f
                isAllCaps = false
                background = if (t == tone) keyBackground(ACCENT_BLUE) else keyBackground(KEY_BG)
                setTextColor(if (t == tone) ON_ACCENT else TEXT)
                stateListAnimator = null
                layoutParams = android.widget.GridLayout.LayoutParams().apply {
                    width = px(94)
                    height = px(44)
                    setMargins(px(4), px(4), px(4), px(4))
                }
                setOnClickListener { v ->
                    v.performHapticFeedback(android.view.HapticFeedbackConstants.KEYBOARD_TAP)
                    tone = t
                    updateToneChip()
                    ChatStore.packageName.takeIf { it.isNotEmpty() }?.let {
                        Prefs.setToneForApp(this@AiKeyboardService, it, t)
                    }
                    dismissPopup()
                }
            })
        }
        popup = PopupWindow(
            grid, LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT, true
        ).apply {
            isOutsideTouchable = true
            setBackgroundDrawable(ColorDrawable(0x00000000))
            showAsDropDown(anchor, 0, px(4))
        }
    }

    private fun resetSuggestions(placeholder: String? = null) {
        stopGenTimer()
        if (!::suggestionButtons.isInitialized) return
        generateBtn.visibility = View.VISIBLE
        generateBtn.text = placeholder ?: "✦  Generate AI replies"
        suggestionButtons.forEach { it.visibility = View.GONE }
        if (::tipView.isInitialized) tipView.visibility = View.GONE
    }

    /** Shows the generated replies as full-width cards + the 💡 tip. */
    private fun showReplies(replies: List<String>, mood: String, trap: String?) {
        if (!::suggestionButtons.isInitialized) return
        generateBtn.visibility = View.GONE
        suggestionButtons.forEachIndexed { i, b ->
            val r = replies.getOrNull(i)
            if (r == null) {
                b.visibility = View.GONE
                return@forEachIndexed
            }
            b.text = r
            b.visibility = View.VISIBLE
        }
        if (::tipView.isInitialized) {
            tipView.text = "💡 ${AdviceKB.tip(mood, trap)}"
            tipView.visibility = View.VISIBLE
        }
    }

    private fun insertSuggestion(text: String, index: Int) {
        if (text.isBlank()) return
        currentInputConnection?.commitText(text, 1)
        resetSuggestions()
    }

    private fun rewriteDraft() {
        val ic = currentInputConnection ?: return
        val selected = ic.getSelectedText(0)?.toString()
        val draft = selected ?: ic.getTextBeforeCursor(2000, 0)?.toString().orEmpty()
        if (draft.isBlank()) {
            Toast.makeText(this, "Type or select a draft first, then tap ✎", Toast.LENGTH_SHORT).show()
            return
        }
        resetSuggestions("…")
        LlmClient.rewriteText(this, tone, draft) { result ->
            result.fold(
                onSuccess = { text ->
                    val conn = currentInputConnection ?: return@fold
                    if (selected != null) {
                        conn.commitText(text, 1)
                    } else {
                        // deleteSurroundingText counts code points, not Java chars —
                        // counting wrong corrupts drafts containing emoji.
                        val codePoints = draft.codePointCount(0, draft.length)
                        conn.deleteSurroundingText(codePoints, 0)
                        conn.commitText(text, 1)
                    }
                    resetSuggestions()
                },
                onFailure = { e ->
                    resetSuggestions()
                    Toast.makeText(this, e.message ?: "Rewrite failed", Toast.LENGTH_LONG).show()
                }
            )
        }
    }

    /** 📸 Live capture: consent dialog → one-shot screen grab → ✦ analyzes image + text. */
    private fun startLiveCapture() {
        val pkg = ChatStore.packageName
        if (pkg.isNotEmpty() && Prefs.excludedApps(this).any { it.equals(pkg, ignoreCase = true) }) {
            Toast.makeText(this, "Screen capture is disabled for this app (privacy)", Toast.LENGTH_LONG).show()
            return
        }
        startActivity(
            android.content.Intent(this, com.aireply.keyboard.ui.CaptureConsentActivity::class.java)
                .addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
        )
    }

    private fun generateReplies() {
        // Always refresh the snapshot from the current windows before generating,
        // so newly received messages are included.
        com.aireply.keyboard.reader.ChatReaderService.instance?.captureNow()
        var snapshot = ChatStore.latestText
        val image = ChatStore.latestImage
            ?.takeIf { System.currentTimeMillis() - ChatStore.latestImageAt < 120_000 }
        if (snapshot.isEmpty() && image == null) {
            val why = when {
                !com.aireply.keyboard.reader.ChatReaderService.isRunning ->
                    "Screen reader is off — enable it in Settings → Accessibility (see app setup)."
                ChatStore.updatedAt == 0L ->
                    "No screen text captured. Open the chat first, keep it open, then tap ✦."
                else ->
                    "Captured text is stale (${(System.currentTimeMillis() - ChatStore.updatedAt) / 1000}s old). Reopen the chat, then tap ✦."
            }
            Toast.makeText(this, why, Toast.LENGTH_LONG).show()
            return
        }
        resetSuggestions("…")
        val read = MoodSensor.read(snapshot)
        // Tapping ✦ again on the same screen = regenerate: ask for different replies.
        val avoid = if (snapshot == lastSnapshot) lastReplies else emptyList()
        lastSnapshot = snapshot
        val done: (Result<Pair<String?, List<String>>>) -> Unit = { result ->
            stopGenTimer()
            result.fold(
                onSuccess = { (detected, replies) ->
                    lastReplies = replies
                    if (::suggestionButtons.isInitialized) {
                        suggestionButtons.forEachIndexed { i, b -> b.text = replies.getOrElse(i) { "—" } }
                    }
                    showReplies(replies, read.mood, read.trap)
                    if (::tipView.isInitialized && tipView.visibility == View.VISIBLE) {
                        tipView.append("   ·  ⚡ ${genSeconds()}")
                    }
                    detected?.let {
                        Toast.makeText(this, "Analyzed: $it", Toast.LENGTH_SHORT).show()
                    }
                },
                onFailure = { e ->
                    lastReplies = emptyList()
                    if (::suggestionButtons.isInitialized) {
                        val offline = OfflineFallback.suggest(tone, read.mood, Prefs.bilingual(this))
                        suggestionButtons.forEachIndexed { i, b -> b.text = offline.getOrElse(i) { "—" } }
                        if (::generateBtn.isInitialized) {
                            generateBtn.visibility = View.VISIBLE
                            generateBtn.text = "✦  Retry"
                        }
                    }
                    Toast.makeText(
                        this,
                        e.message?.takeIf { !it.contains("API error") } ?: "Offline suggestions — API unreachable. Check settings/connection.",
                        Toast.LENGTH_LONG
                    ).show()
                }
            )
        }
        if (image != null) {
            LlmClient.generateFromLiveScreen(this, tone, image, snapshot.ifEmpty { "(no readable text)" }, avoid, done)
        } else {
            LlmClient.generate(this, tone, snapshot, avoid) { result ->
                stopGenTimer()
                result.fold(
                    onSuccess = { replies ->
                        lastReplies = replies
                        if (::suggestionButtons.isInitialized) {
                            suggestionButtons.forEachIndexed { i, b -> b.text = replies.getOrElse(i) { "—" } }
                        }
                        showReplies(replies, read.mood, read.trap)
                        if (::tipView.isInitialized && tipView.visibility == View.VISIBLE) {
                            tipView.append("   ·  ⚡ ${genSeconds()}")
                        }
                    },
                    onFailure = {
                        lastReplies = emptyList()
                        if (::suggestionButtons.isInitialized) {
                            val offline = OfflineFallback.suggest(tone, read.mood, Prefs.bilingual(this))
                            suggestionButtons.forEachIndexed { i, b -> b.text = offline.getOrElse(i) { "—" } }
                            if (::generateBtn.isInitialized) {
                                generateBtn.visibility = View.VISIBLE
                                generateBtn.text = "✦  Retry"
                            }
                        }
                        Toast.makeText(
                            this,
                            "Offline suggestions — API unreachable. Check settings/connection.",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                )
            }
        }
        startGenTimer()
        read.trap?.let { trap ->
            Toast.makeText(this, "⚠️ $trap: ${read.trapAdvice}", Toast.LENGTH_LONG).show()
        }
    }

    private fun rebuildKeys(px: (Int) -> Int) {
        keysColumn.removeAllViews()
        val hs = Prefs.kbScale(this) // keyboard height scale from settings
        fun rowH() = px((46 * hs).toInt())
        fun keyH() = px((44 * hs).toInt())

        if (emojiOn) {
            emojiRows.forEach { chars ->
                val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
                chars.forEach { e -> row.addView(makeEmojiKey(e, px)) }
                keysColumn.addView(row, LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, rowH()
                ))
            }
        } else {
            // Gboard structure: row3 = [⇧ or page-toggle] + 7 keys + [⌫]
            val rows: List<List<Char>> = when {
                !symbolsOn -> accentRows
                else -> symbolPages[symbolPage]
            }
            rows.forEachIndexed { rowIndex, chars ->
                val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
                if (rowIndex == 2) {
                    if (!symbolsOn) {
                        row.addView(makeSpecialKey(if (shiftOn) "⇪" else "⇧", px, weight = 1.4f) {
                            shiftOn = !shiftOn
                            rebuildKeys(px)
                        })
                    } else {
                        // page toggle: =\< on page 0, back to numbers on page 1
                        row.addView(makeSpecialKey(if (symbolPage == 0) "=\\<" else "123", px, weight = 1.4f) {
                            symbolPage = if (symbolPage == 0) 1 else 0
                            rebuildKeys(px)
                        })
                    }
                }
                chars.forEach { c ->
                    row.addView(makeCharKey(c, px))
                }
                if (rowIndex == 2) {
                    row.addView(makeSpecialKey("⌫", px, weight = 1.4f) {
                        sendDownUpKeyEvents(KeyEvent.KEYCODE_DEL)
                    })
                }
                keysColumn.addView(row, LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, rowH()
                ))
            }
        }

        // Gboard bottom row: [😊] [?123] [,] [space] [.] [↵]
        val bottom = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        bottom.addView(makeSpecialKey("😊", px, weight = 1f) {
            emojiOn = !emojiOn
            rebuildKeys(px)
        })
        bottom.addView(makeSpecialKey(if (symbolsOn) "ABC" else "?123", px, weight = 1.4f) {
            symbolsOn = !symbolsOn
            symbolPage = 0
            rebuildKeys(px)
        })
        bottom.addView(makeCharKey(',', px, weight = 1f))
        // Space bar carries the keyboard branding, Gboard-style.
        bottom.addView(makeSpecialKey("AI Reply", px, weight = 4f) {
            currentInputConnection?.commitText(" ", 1)
        })
        bottom.addView(makeCharKey('.', px, weight = 1f))
        bottom.addView(makeSpecialKey("↵", px, weight = 1.4f) { commitEnter() })
        keysColumn.addView(bottom, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, rowH()
        ))
    }

    private fun makeCharKey(c: Char, px: (Int) -> Int, weight: Float = 1f): Button {
        val label = when {
            symbolsOn -> c.toString()
            shiftOn -> c.uppercaseChar().toString()
            else -> c.toString()
        }
        return Button(this).apply {
            text = label
            textSize = 17f
            background = keyBackground(KEY_BG)
            setTextColor(TEXT)
            stateListAnimator = null
            layoutParams = LinearLayout.LayoutParams(0, px(44), weight).apply { setMargins(px(2), px(2), px(2), px(2)) }
            setOnClickListener {
                performHapticFeedback(android.view.HapticFeedbackConstants.KEYBOARD_TAP)
                currentInputConnection?.commitText(label, 1)
                if (shiftOn && !symbolsOn) {
                    shiftOn = false
                    rebuildKeys(px)
                }
            }
            // Long-press a letter for accented variants (é, ñ, ü, …).
            if (!symbolsOn) {
                val base = c.lowercaseChar()
                ACCENTS[base]?.let { variants ->
                    setOnLongClickListener {
                        showPopup(this, if (shiftOn) variants.map { it.uppercaseChar().toString() }
                        else variants.map { it.toString() }, px)
                        true
                    }
                }
            }
        }
    }

    private fun makeEmojiKey(emoji: String, px: (Int) -> Int): Button {
        return Button(this).apply {
            text = emoji
            textSize = 18f
            background = keyBackground(KEY_BG)
            stateListAnimator = null
            layoutParams = LinearLayout.LayoutParams(0, px(44), 1f).apply { setMargins(px(2), px(2), px(2), px(2)) }
            setOnClickListener { v ->
                v.performHapticFeedback(android.view.HapticFeedbackConstants.KEYBOARD_TAP)
                currentInputConnection?.commitText(emoji, 1)
            }
        }
    }

    /** Small flyout above a key with tappable variants (accents / symbols). */
    private fun showPopup(anchor: View, variants: List<String>, px: (Int) -> Int) {
        dismissPopup()
        val location = IntArray(2)
        anchor.getLocationOnScreen(location)
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            background = ColorDrawable(0xEE1A1B20.toInt())
            setPadding(px(6), px(6), px(6), px(6))
        }
        variants.forEach { v ->
            row.addView(Button(this).apply {
                text = v
                textSize = 18f
                isAllCaps = false
                background = keyBackground(KEY_BG)
                setTextColor(TEXT)
                stateListAnimator = null
                layoutParams = LinearLayout.LayoutParams(px(52), px(52)).apply { setMargins(px(2), 0, px(2), 0) }
                setOnClickListener {
                    currentInputConnection?.commitText(v, 1)
                    dismissPopup()
                }
            })
        }
        popup = PopupWindow(row, LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT, false).apply {
            isOutsideTouchable = true
            setBackgroundDrawable(ColorDrawable(0x00000000))
            showAtLocation(anchor, Gravity.NO_GRAVITY, location[0], location[1] - px(60))
        }
    }

    private fun dismissPopup() {
        popup?.dismiss()
        popup = null
    }

    private fun makeSpecialKey(label: String, px: (Int) -> Int, weight: Float, action: () -> Unit): Button {
        return Button(this).apply {
            text = label
            textSize = if (label.length > 2) 13f else 16f
            isAllCaps = false
            // Gboard: enter and active shift are the accent blue; other keys are muted.
            val accent = label == "↵" || label == "⇪"
            background = keyBackground(if (accent) ACCENT_BLUE else SPECIAL_BG)
            setTextColor(if (accent) ON_ACCENT else TEXT_DIM)
            stateListAnimator = null
            layoutParams = LinearLayout.LayoutParams(0, px(44), weight).apply { setMargins(px(2), px(2), px(2), px(2)) }
            // Long-press the spacebar (Gboard behavior) to switch keyboards.
            if (label == "AI Reply") {
                setOnLongClickListener { v ->
                    v.performHapticFeedback(android.view.HapticFeedbackConstants.LONG_PRESS)
                    val imm = this@AiKeyboardService.getSystemService(INPUT_METHOD_SERVICE)
                        as android.view.inputmethod.InputMethodManager
                    imm.showInputMethodPicker()
                    true
                }
            }
            setOnClickListener { v ->
                v.performHapticFeedback(android.view.HapticFeedbackConstants.KEYBOARD_TAP)
                action()
            }
        }
    }

    private fun commitEnter() {
        val ic = currentInputConnection ?: return
        val ei = currentInputEditorInfo
        val action = ei?.actionId ?: EditorInfo.IME_ACTION_UNSPECIFIED
        if (ei != null && ei.inputType != EditorInfo.TYPE_NULL &&
            action != EditorInfo.IME_ACTION_UNSPECIFIED && action != EditorInfo.IME_ACTION_NONE
        ) {
            ic.performEditorAction(action)
        } else {
            sendDownUpKeyEvents(KeyEvent.KEYCODE_ENTER)
        }
    }

    override fun onEvaluateFullscreenMode(): Boolean = false

    override fun onStartInputView(info: EditorInfo?, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        // Rebuild the view if the user changed theme or keyboard height in settings.
        val cfg = Prefs.themeIndex(this) * 10 + Prefs.kbScaleIndex(this)
        if (cfg != builtConfig) {
            applyTheme()
            setInputView(onCreateInputView())
            builtConfig = cfg
        }
        // Restore the tone remembered for this chat app, if any; keep chosen tone otherwise.
        ChatStore.packageName.takeIf { it.isNotEmpty() }
            ?.let { Prefs.toneForApp(this, it) }
            ?.let { tone = it }
        updateToneChip()
        resetSuggestions()
    }
}
