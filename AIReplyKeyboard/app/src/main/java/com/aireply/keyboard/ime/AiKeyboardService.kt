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
import com.aireply.keyboard.ai.LlmClient
import com.aireply.keyboard.ai.MoodSensor
import com.aireply.keyboard.ai.OfflineFallback
import com.aireply.keyboard.ai.Prefs
import com.aireply.keyboard.ai.Tone
import com.aireply.keyboard.reader.ChatStore

class AiKeyboardService : InputMethodService() {

    private lateinit var root: LinearLayout
    private lateinit var keysColumn: LinearLayout
    private lateinit var chipRow: LinearLayout
    private lateinit var suggestionButtons: List<Button>

    private var shiftOn = false
    private var symbolsOn = false
    private var emojiOn = false
    private var tone: Tone = Tone.CASUAL
    private var popup: PopupWindow? = null

    private var lastSnapshot: String = ""
    private var lastReplies: List<String> = emptyList()

    private val emojiRows = listOf(
        listOf("😀", "😂", "🥲", "😍", "😘", "😜", "🤔", "😏", "😎", "🥳", "😇"),
        listOf("😢", "😭", "😡", "🙄", "😴", "🤗", "🤩", "🫣", "🤠", "😈", "👀"),
        listOf("👍", "👏", "🙌", "👌", "✌️", "🤞", "🫶", "❤️", "💔", "🔥", "💯"),
        listOf("✨", "🎉", "🌹", "☕", "🍕", "🎮", "⚽", "🎧", "📸", "💀", "🤡")
    )

    private val accentRows = listOf(
        listOf('q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'),
        listOf('a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'),
        listOf('z', 'x', 'c', 'v', 'b', 'n', 'm')
    )
    private val symbolRows = listOf(
        listOf('1', '2', '3', '4', '5', '6', '7', '8', '9', '0'),
        listOf('@', '#', '$', '%', '&', '-', '+', '(', ')'),
        listOf('=', '*', '"', '\'', ':', ';', '!', '?')
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
        tone = Prefs.defaultTone(this)
    }

    override fun onDestroy() {
        if (instance === this) instance = null
        dismissPopup()
        super.onDestroy()
    }

    /** Debug test hooks — driven by TestHookReceiver in debug builds only. */
    fun debugGenerate() = generateReplies()

    fun debugInsert(index: Int) {
        val text = lastReplies.getOrElse(index) { return }
        insertSuggestion(text, index)
    }

    override fun onCreateInputView(): View {
        val density = resources.displayMetrics.density
        val px: (Int) -> Int = { dp -> (dp * density).toInt() }

        root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xFF191C20.toInt())
            setPadding(px(6), px(4), px(6), px(6))
        }

        // --- Suggestion row: three AI replies ---
        val suggestionRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        suggestionButtons = (0 until 3).map { index ->
            Button(this).apply {
                background = getDrawableSafe(com.aireply.keyboard.R.drawable.bg_suggestion)
                setTextColor(0xFFE8EAED.toInt())
                textSize = 13f
                isAllCaps = false
                setSingleLine()
                ellipsize = TextUtils.TruncateAt.END
                setPadding(px(8), px(6), px(8), px(6))
                gravity = Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(0, px(40), 1f).apply {
                    setMargins(px(3), 0, px(3), 0)
                }
                setOnClickListener { insertSuggestion(text.toString(), index) }
                setOnLongClickListener {
                    val t = text.toString()
                    if (t.startsWith("Tap ✦") || t == "…" || t.startsWith("Tap ✦ to retry")) {
                        false
                    } else {
                        Prefs.addSaved(this@AiKeyboardService, t)
                        Toast.makeText(this@AiKeyboardService, "Saved 🔖 — find it in the app", Toast.LENGTH_SHORT).show()
                        true
                    }
                }
            }.also { suggestionRow.addView(it) }
        }
        resetSuggestions()
        root.addView(suggestionRow)

        // --- Tone chips + generate button ---
        val toneRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val chipScroll = HorizontalScrollView(this).apply {
            isHorizontalScrollBarEnabled = false
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        chipRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        Tone.entries.forEach { t ->
            chipRow.addView(makeChip(t, px))
        }
        chipScroll.addView(chipRow)
        toneRow.addView(chipScroll)

        toneRow.addView(Button(this).apply {
            text = "🌐"
            textSize = 16f
            background = getDrawableSafe(com.aireply.keyboard.R.drawable.bg_key_special)
            setTextColor(0xFFBDC1C6.toInt())
            layoutParams = LinearLayout.LayoutParams(px(48), px(40)).apply { setMargins(px(4), 0, 0, 0) }
            setOnClickListener { switchToNextInputMethod(false) }
        })
        toneRow.addView(Button(this).apply {
            text = "✎"
            textSize = 16f
            background = getDrawableSafe(com.aireply.keyboard.R.drawable.bg_key_special)
            setTextColor(0xFFE8C468.toInt())
            layoutParams = LinearLayout.LayoutParams(px(48), px(40)).apply { setMargins(px(4), 0, 0, 0) }
            setOnClickListener { rewriteDraft() }
        })
        toneRow.addView(Button(this).apply {
            text = "✦"
            textSize = 18f
            background = getDrawableSafe(com.aireply.keyboard.R.drawable.bg_key_special)
            setTextColor(0xFF7FE0C8.toInt())
            layoutParams = LinearLayout.LayoutParams(px(48), px(40)).apply { setMargins(px(4), 0, 0, 0) }
            setOnClickListener { generateReplies() }
        })
        root.addView(toneRow, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { topMargin = px(6) })

        // --- QWERTY keys ---
        keysColumn = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        root.addView(keysColumn, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { topMargin = px(6) })

        rebuildKeys(px)
        return root
    }

    private fun makeChip(t: Tone, px: (Int) -> Int): Button {
        val selected = t == tone
        return Button(this).apply {
            text = t.label
            tag = t
            textSize = 12f
            isAllCaps = false
            background = getDrawableSafe(
                if (selected) com.aireply.keyboard.R.drawable.bg_chip_selected
                else com.aireply.keyboard.R.drawable.bg_chip
            )
            setTextColor(if (selected) 0xFFFFFFFF.toInt() else 0xFFBDC1C6.toInt())
            setSingleLine()
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, px(34)
            ).apply { setMargins(px(3), 0, px(3), 0) }
            setPadding(px(12), 0, px(12), 0)
            setOnClickListener {
                tone = t
                styleChips()
                // Remember the vibe for this chat app so reopening it feels the same.
                ChatStore.packageName.takeIf { it.isNotEmpty() }?.let {
                    Prefs.setToneForApp(this@AiKeyboardService, it, t)
                }
            }
        }
    }

    /** Restyles chips in place — no view rebuild, suggestions stay put. */
    private fun styleChips() {
        for (i in 0 until chipRow.childCount) {
            val b = chipRow.getChildAt(i) as? Button ?: continue
            val t = b.tag as? Tone ?: continue
            val selected = t == tone
            b.background = getDrawableSafe(
                if (selected) com.aireply.keyboard.R.drawable.bg_chip_selected
                else com.aireply.keyboard.R.drawable.bg_chip
            )
            b.setTextColor(if (selected) 0xFFFFFFFF.toInt() else 0xFFBDC1C6.toInt())
        }
    }

    private fun resetSuggestions(placeholder: String? = null) {
        val hints = placeholder?.let { listOf(it, it, it) } ?: listOf("Tap ✦ for reply 1", "Tap ✦ for reply 2", "Tap ✦ for reply 3")
        suggestionButtons.forEachIndexed { i, b -> b.text = hints[i] }
    }

    private fun insertSuggestion(text: String, index: Int) {
        if (text.startsWith("Tap ✦") || text == "…") return
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

    private fun generateReplies() {
        val snapshot = ChatStore.latestText
        if (snapshot.isEmpty()) {
            Toast.makeText(this, "No chat text captured yet — make sure Accessibility is enabled for AI Reply.", Toast.LENGTH_LONG).show()
            return
        }
        resetSuggestions("…")
        val read = MoodSensor.read(snapshot)
        // Tapping ✦ again on the same screen = regenerate: ask for different replies.
        val avoid = if (snapshot == lastSnapshot) lastReplies else emptyList()
        lastSnapshot = snapshot
        LlmClient.generate(this, tone, snapshot, avoid) { result ->
            result.fold(
                onSuccess = { replies ->
                    lastReplies = replies
                    suggestionButtons.forEachIndexed { i, b -> b.text = replies.getOrElse(i) { "—" } }
                },
                onFailure = {
                    lastReplies = emptyList()
                    val offline = OfflineFallback.suggest(tone, read.mood, Prefs.bilingual(this))
                    suggestionButtons.forEachIndexed { i, b -> b.text = offline.getOrElse(i) { "—" } }
                    Toast.makeText(
                        this,
                        "Offline suggestions — API unreachable. Check settings/connection.",
                        Toast.LENGTH_LONG
                    ).show()
                }
            )
        }
        read.trap?.let { trap ->
            Toast.makeText(this, "⚠️ $trap: ${read.trapAdvice}", Toast.LENGTH_LONG).show()
        }
    }

    private fun rebuildKeys(px: (Int) -> Int) {
        keysColumn.removeAllViews()

        if (emojiOn) {
            emojiRows.forEach { chars ->
                val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
                chars.forEach { e -> row.addView(makeEmojiKey(e, px)) }
                keysColumn.addView(row, LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, px(46)
                ))
            }
        } else {
            val rows = if (symbolsOn) symbolRows else accentRows
            rows.forEachIndexed { rowIndex, chars ->
                val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
                if (rowIndex == 2) {
                    row.addView(makeSpecialKey(if (symbolsOn) "ABC" else "123", px, weight = 1.4f) {
                        symbolsOn = !symbolsOn
                        rebuildKeys(px)
                    })
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
                    LinearLayout.LayoutParams.MATCH_PARENT, px(46)
                ))
            }
        }

        // Bottom row: shift/ABC, comma, space, emoji, period, enter
        val bottom = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        bottom.addView(makeSpecialKey(
            when {
                emojiOn -> "ABC"
                shiftOn -> "⇪"
                else -> "⇧"
            }, px, weight = 1.4f
        ) {
            if (emojiOn) {
                emojiOn = false
            } else {
                shiftOn = !shiftOn
            }
            rebuildKeys(px)
        })
        bottom.addView(makeCharKey(',', px, weight = 1f))
        bottom.addView(makeSpecialKey("space", px, weight = 3.5f) {
            currentInputConnection?.commitText(" ", 1)
        })
        bottom.addView(makeSpecialKey("😊", px, weight = 1f) {
            emojiOn = !emojiOn
            rebuildKeys(px)
        })
        bottom.addView(makeCharKey('.', px, weight = 1f))
        bottom.addView(makeSpecialKey("↵", px, weight = 1.4f) { commitEnter() })
        keysColumn.addView(bottom, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, px(46)
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
            textSize = 16f
            background = getDrawableSafe(com.aireply.keyboard.R.drawable.bg_key)
            setTextColor(0xFFE8EAED.toInt())
            stateListAnimator = null
            layoutParams = LinearLayout.LayoutParams(0, px(44), weight).apply { setMargins(px(2), px(2), px(2), px(2)) }
            setOnClickListener {
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
            background = getDrawableSafe(com.aireply.keyboard.R.drawable.bg_key)
            stateListAnimator = null
            layoutParams = LinearLayout.LayoutParams(0, px(44), 1f).apply { setMargins(px(2), px(2), px(2), px(2)) }
            setOnClickListener { currentInputConnection?.commitText(emoji, 1) }
        }
    }

    /** Small flyout above a key with tappable variants (accents / symbols). */
    private fun showPopup(anchor: View, variants: List<String>, px: (Int) -> Int) {
        dismissPopup()
        val location = IntArray(2)
        anchor.getLocationOnScreen(location)
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            background = ColorDrawable(0xFF2A2D31.toInt())
            setPadding(px(4), px(4), px(4), px(4))
        }
        variants.forEach { v ->
            row.addView(Button(this).apply {
                text = v
                textSize = 18f
                isAllCaps = false
                background = getDrawableSafe(com.aireply.keyboard.R.drawable.bg_key_special)
                setTextColor(0xFFE8EAED.toInt())
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
            textSize = if (label.length > 2) 12f else 16f
            isAllCaps = false
            background = getDrawableSafe(com.aireply.keyboard.R.drawable.bg_key_special)
            setTextColor(0xFFBDC1C6.toInt())
            stateListAnimator = null
            layoutParams = LinearLayout.LayoutParams(0, px(44), weight).apply { setMargins(px(2), px(2), px(2), px(2)) }
            setOnClickListener { action() }
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

    private fun getDrawableSafe(id: Int) = getDrawable(id)!!

    override fun onEvaluateFullscreenMode(): Boolean = false

    override fun onStartInputView(info: EditorInfo?, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        // Restore the tone remembered for this chat app, if any; keep chosen tone otherwise.
        ChatStore.packageName.takeIf { it.isNotEmpty() }
            ?.let { Prefs.toneForApp(this, it) }
            ?.let { tone = it }
        if (::chipRow.isInitialized) styleChips()
        resetSuggestions()
    }
}
