package com.aireply.keyboard.reader

import android.accessibilityservice.AccessibilityService
import android.os.Handler
import android.os.Looper
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import com.aireply.keyboard.ai.Prefs

/**
 * Captures the visible text of whatever app is in the foreground and stores the latest
 * snapshot in [ChatStore]. The keyboard reads that snapshot when the user taps "generate".
 */
class ChatReaderService : AccessibilityService() {

    private val handler = Handler(Looper.getMainLooper())
    private var pending: Runnable? = null

    override fun onServiceConnected() {
        instance = this
    }

    override fun onDestroy() {
        if (instance === this) instance = null
        super.onDestroy()
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        val e = event ?: return
        val pkg = e.packageName?.toString() ?: return
        if (pkg == applicationContext.packageName) return
        // Privacy: never capture content from excluded apps, and drop any earlier snapshot
        // so stale text isn't used while the user is inside an excluded app.
        if (Prefs.excludedApps(applicationContext).any { it.equals(pkg, ignoreCase = true) }) {
            pending?.let { handler.removeCallbacks(it) }
            ChatStore.clear()
            return
        }
        val relevant = e.eventType == AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED ||
            e.eventType == AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED
        if (!relevant) return

        // Debounce: chat apps fire many content-change events per keystroke/animation.
        pending?.let { handler.removeCallbacks(it) }
        val capture = Runnable { captureSnapshot(pkg) }
        pending = capture
        handler.postDelayed(capture, 300)
    }

    private fun captureSnapshot(pkg: String) {
        val root: AccessibilityNodeInfo = rootInActiveWindow ?: return
        val sb = StringBuilder()
        collectText(root, sb, 0)
        val text = sb.toString().trim()
        if (text.isNotEmpty()) ChatStore.update(pkg, text)
    }

    private fun collectText(node: AccessibilityNodeInfo, sb: StringBuilder, depth: Int) {
        if (depth > 40 || sb.length > 8000) return
        node.text?.toString()?.trim()?.takeIf { it.isNotEmpty() }?.let { sb.appendLine(it) }
        for (i in 0 until node.childCount) {
            val child = node.getChild(i) ?: continue
            collectText(child, sb, depth + 1)
            child.recycle()
        }
    }

    override fun onInterrupt() = Unit

    companion object {
        @JvmStatic
        @Volatile
        var instance: ChatReaderService? = null
            private set

        val isRunning: Boolean get() = instance != null
    }
}
