package com.aireply.keyboard.debug

import android.app.Activity
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/**
 * DEBUG-ONLY chat simulator: a scripted "her" persona to demo and test the
 * full capture → generate → insert → send loop without WhatsApp.
 * Launched via: adb shell am start -n com.aireply.keyboard/.debug.ChatSimActivity
 */
class ChatSimActivity : Activity() {

    private val herLines = listOf(
        "haha hey! saw your gym story 😄",
        "omg yes I'm obsessed with that dog",
        "lol okay you get one question 😏",
        "hmm coffee person... chai actually ☕",
        "wow smooth. maybe Saturday?",
        "bet. send me the place 😌"
    )
    private var herIdx = 0

    private fun px(dp: Int) = (dp * resources.displayMetrics.density).toInt()

    private lateinit var bubbles: LinearLayout
    private lateinit var scroll: ScrollView
    private lateinit var input: EditText
    private val handler = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = "ChatSim"

        bubbles = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(px(12), px(12), px(12), px(12))
        }
        scroll = ScrollView(this).apply { addView(bubbles) }

        input = EditText(this).apply {
            hint = "Type a message…"
            setSingleLine()
        }
        val send = Button(this).apply { text = "Send ▶" }

        val composer = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(input, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
            addView(send, LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT))
        }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(scroll, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
            addView(composer, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
        }
        setContentView(root)

        // She opens the conversation.
        handler.postDelayed({ herSays() }, 600)

        send.setOnClickListener {
            val text = input.text.toString().trim()
            if (text.isEmpty()) return@setOnClickListener
            addBubble("You", text, 0xFF2E7D6B.toInt())
            input.setText("")
            handler.postDelayed({ herSays() }, 900)
        }
    }

    private fun herSays() {
        if (herIdx >= herLines.size) {
            addBubble("Her", "(scripted convo complete — reset the demo to replay)", 0xFF616161.toInt())
            return
        }
        addBubble("Her", herLines[herIdx], 0xFF3C4043.toInt())
        herIdx++
        scroll.post { scroll.fullScroll(ScrollView.FOCUS_DOWN) }
    }

    private fun addBubble(who: String, text: String, color: Int) {
        val tv = TextView(this).apply {
            this.text = "$who: $text"
            setTextColor(0xFFFFFFFF.toInt())
            background = android.graphics.drawable.GradientDrawable().apply {
                setColor(color)
                cornerRadius = px(14).toFloat()
            }
            setPadding(px(14), px(10), px(14), px(10))
            textSize = 15f
        }
        val wrap = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val lp = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
            setMargins(px(4), px(4), px(4), px(4))
        }
        if (who == "You") wrap.gravity = Gravity.END
        wrap.addView(tv, lp)
        bubbles.addView(wrap)
        scroll.post { scroll.fullScroll(ScrollView.FOCUS_DOWN) }
    }
}
