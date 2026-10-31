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
        android.util.Log.i("ChatReader", "screen reader connected")
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
        val root = windowRootFor(pkg) ?: return
        val sb = StringBuilder()
        collectText(root, sb, 0)
        val text = sb.toString().trim()
        if (text.isNotEmpty()) {
            ChatStore.update(pkg, text)
            android.util.Log.d("ChatReader", "captured ${text.length} chars from $pkg")
        }
    }

    /**
     * Root of the given app's window. Critical: while an IME is open,
     * rootInActiveWindow often IS the keyboard — so prefer the app's own window
     * from the window list, falling back to the active window.
     */
    private fun windowRootFor(pkg: String): AccessibilityNodeInfo? {
        windows?.firstOrNull { w ->
            w.root?.packageName?.toString() == pkg
        }?.root?.let { return it }
        return rootInActiveWindow?.takeIf { it.packageName?.toString() == pkg }
    }

    /**
     * On-demand capture for the ✦ button: grabs text from the most likely chat
     * window even if no accessibility event was processed. Returns chars captured.
     */
    fun captureNow(): Int {
        val excluded = Prefs.excludedApps(applicationContext)
        var node: AccessibilityNodeInfo? = null
        var nodePkg = ""
        windows?.forEach { w ->
            if (node != null) return@forEach
            val root = w.root ?: return@forEach
            val wpkg = root.packageName?.toString() ?: return@forEach
            val isSelf = wpkg == applicationContext.packageName
            // Our own windows are off-limits EXCEPT the debug ChatSimulator
            // (title "ChatSim"), which exists to demo the pipeline.
            val isSim = isSelf && w.title?.toString()?.contains("ChatSim") == true
            if (isSelf && !isSim) return@forEach
            if (!isSelf && wpkg == "com.android.systemui") return@forEach
            if (excluded.any { it.equals(wpkg, ignoreCase = true) } && !isSim) return@forEach
            node = root
            nodePkg = wpkg
        }
        val target = node ?: rootInActiveWindow?.takeIf {
            it.packageName?.toString() != applicationContext.packageName &&
                !excluded.any { e -> it.packageName?.toString()?.equals(e, true) == true }
        } ?: return 0
        val sb = StringBuilder()
        collectText(target, sb, 0)
        val text = sb.toString().trim()
        if (text.isNotEmpty()) {
            ChatStore.update(nodePkg.ifEmpty { target.packageName?.toString() ?: "" }, text)
        }
        return text.length
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
