package com.aireply.keyboard.ui

import android.accessibilityservice.AccessibilityServiceInfo
import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.accessibility.AccessibilityManager
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.TextView
import com.aireply.keyboard.R
import com.aireply.keyboard.reader.ChatReaderService

class MainActivity : Activity() {

    private lateinit var tvStatus: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvStatus = findViewById(R.id.tvStatus)
        findViewById<Button>(R.id.btnSettings).setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }
        findViewById<Button>(R.id.btnAnalyzer).setOnClickListener {
            startActivity(Intent(this, AnalyzerActivity::class.java))
        }
        findViewById<Button>(R.id.btnLibrary).setOnClickListener {
            startActivity(Intent(this, LibraryActivity::class.java))
        }
        findViewById<Button>(R.id.btnSaved).setOnClickListener {
            startActivity(Intent(this, SavedActivity::class.java))
        }
    }

    override fun onResume() {
        super.onResume()
        val keyboardOn = (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager)
            .enabledInputMethodList.any { it.packageName == packageName }
        val accOn = (getSystemService(ACCESSIBILITY_SERVICE) as AccessibilityManager)
            .getEnabledAccessibilityServiceList(AccessibilityServiceInfo.FEEDBACK_ALL_MASK)
            .any { it.resolveInfo.serviceInfo.packageName == packageName && it.id.endsWith(ChatReaderService::class.java.simpleName) }

        tvStatus.text = when {
            keyboardOn && accOn -> "✓ Keyboard and screen reader are enabled — tap ✦ in any chat to get replies."
            keyboardOn -> "Keyboard enabled. Still needed: enable the screen reader in Settings."
            accOn -> "Screen reader enabled. Still needed: enable the AI Reply keyboard."
            else -> "Setup needed: enable the AI Reply keyboard and the screen reader."
        }
        tvStatus.visibility = android.view.View.VISIBLE
    }
}
