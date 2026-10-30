package com.aireply.keyboard.ime

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import com.aireply.keyboard.BuildConfig
import com.aireply.keyboard.ai.Prefs
import com.aireply.keyboard.ai.ProviderPresets
import com.aireply.keyboard.ai.Tone

/**
 * DEBUG-ONLY test hook: lets adb drive the keyboard on an emulator where injected
 * taps can't reach IME windows. No-ops in release builds.
 *   adb shell am broadcast -a com.aireply.keyboard.TEST --es cmd config|generate|insert
 */
class TestHookReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        android.util.Log.i("TestHook", "received cmd=${intent.getStringExtra("cmd")} instance=${AiKeyboardService.instance}")
        if (!BuildConfig.DEBUG) return
        val service = AiKeyboardService.instance ?: return
        when (intent.getStringExtra("cmd")) {
            "config" -> {
                val preset = ProviderPresets.all.first { it.label.contains("web2api") }
                Prefs.save(context, preset.baseUrl ?: "", "test-key", preset.textModel ?: "", Tone.CASUAL)
            }
            "generate" -> service.debugGenerate()
            "insert" -> service.debugInsert(intent.getIntExtra("index", 0))
        }
    }
}
