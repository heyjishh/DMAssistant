package com.aireply.keyboard.ui

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.provider.Settings
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import android.widget.Toast
import com.aireply.keyboard.R
import com.aireply.keyboard.ai.Prefs
import com.aireply.keyboard.ai.ProviderPresets
import com.aireply.keyboard.ai.Tone

class SettingsActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        val etBaseUrl = findViewById<EditText>(R.id.etBaseUrl)
        val etApiKey = findViewById<EditText>(R.id.etApiKey)
        val etModel = findViewById<EditText>(R.id.etModel)
        val etVisionModel = findViewById<EditText>(R.id.etVisionModel)
        val etStyleProfile = findViewById<EditText>(R.id.etStyleProfile)
        val cbBilingual = findViewById<android.widget.CheckBox>(R.id.cbBilingual)
        val etExcludedApps = findViewById<EditText>(R.id.etExcludedApps)
        val spinnerTone = findViewById<Spinner>(R.id.spinnerTone)

        // Provider presets prefill the connection fields; Custom leaves them as-is.
        val spinnerProvider = findViewById<Spinner>(R.id.spinnerProvider)
        val tvProviderHint = findViewById<android.widget.TextView>(R.id.tvProviderHint)
        val presets = ProviderPresets.all
        spinnerProvider.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, presets.map { it.label }
        )
        spinnerProvider.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: android.view.View?, pos: Int, id: Long) {
                val preset = presets[pos]
                tvProviderHint.text = preset.keyHint
                preset.baseUrl?.let {
                    etBaseUrl.setText(it)
                    etModel.setText(preset.textModel)
                    etVisionModel.setText(preset.visionModel)
                }
            }

            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) = Unit
        }

        etBaseUrl.setText(Prefs.baseUrl(this))
        etApiKey.setText(Prefs.apiKey(this))
        etModel.setText(Prefs.model(this))
        etVisionModel.setText(Prefs.visionModel(this))
        etStyleProfile.setText(Prefs.styleProfile(this))
        cbBilingual.isChecked = Prefs.bilingual(this)
        etExcludedApps.setText(Prefs.excludedApps(this).joinToString(", "))

        val labels = Tone.entries.map { it.label }
        spinnerTone.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, labels
        )
        spinnerTone.setSelection(labels.indexOf(Prefs.defaultTone(this).label))

        findViewById<Button>(R.id.btnEnableKeyboard).setOnClickListener {
            startActivity(Intent(Settings.ACTION_INPUT_METHOD_SETTINGS))
        }
        findViewById<Button>(R.id.btnEnableAccessibility).setOnClickListener {
            startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
        }
        findViewById<Button>(R.id.btnDelhiPreset).setOnClickListener {
            etStyleProfile.setText(DELHI_PRESET)
            Toast.makeText(this, "Delhi Gen-Z preset loaded — tap Save", Toast.LENGTH_LONG).show()
        }
        findViewById<Button>(R.id.btnSave).setOnClickListener {
            val tone = Tone.entries[spinnerTone.selectedItemPosition]
            Prefs.save(
                this,
                baseUrl = etBaseUrl.text.toString(),
                apiKey = etApiKey.text.toString(),
                model = etModel.text.toString(),
                tone = tone
            )
            Prefs.saveAdvanced(
                this,
                visionModel = etVisionModel.text.toString(),
                styleProfile = etStyleProfile.text.toString()
            )
            Prefs.saveExtras(
                this,
                bilingual = cbBilingual.isChecked,
                excludedApps = etExcludedApps.text.toString()
            )
            Toast.makeText(this, R.string.toast_saved, Toast.LENGTH_SHORT).show()
        }
    }

    private val DELHI_PRESET = """
        Delhi Gen-Z, Hinglish mix (kk, yaar, chal na, koi na, thik hai), short punchy sentences.
        Be present-focused: reference the exact reel/message/profile detail being replied to.
        Mirror energy: short reply = short back. Open-ended questions, never yes/no.
        Tease playfully, never mean. Compliment humor/taste/energy, never appearance early on.
        One focused point per message, no question bombardment. If they seem dry, one light text then pause, don't chase.
        No em dashes, no hashtags, no markdown. 1-2 emojis max (😂 👀 😏).
    """.trimIndent()
}
