package com.aireply.keyboard.ui

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Base64
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ImageView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import com.aireply.keyboard.R
import com.aireply.keyboard.ai.LlmClient
import com.aireply.keyboard.ai.Tone
import java.io.ByteArrayOutputStream

class AnalyzerActivity : Activity() {

    private var imageBase64: String? = null
    private var lastResult: String = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_analyzer)

        val spinnerTone = findViewById<Spinner>(R.id.spinnerTone)
        val labels = Tone.entries.map { it.label }
        spinnerTone.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, labels)

        findViewById<Button>(R.id.btnPick).setOnClickListener {
            startActivityForResult(
                Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }, REQ_PICK
            )
        }
        findViewById<Button>(R.id.btnOpeners).setOnClickListener {
            imageBase64?.let { analyze(spinnerTone, LlmClient.AnalysisTask.OPENERS, it) }
                ?: toast("Pick a screenshot first")
        }
        findViewById<Button>(R.id.btnReplies).setOnClickListener {
            imageBase64?.let { analyze(spinnerTone, LlmClient.AnalysisTask.REPLIES, it) }
                ?: toast("Pick a screenshot first")
        }
        findViewById<Button>(R.id.btnAudit).setOnClickListener {
            if (imageBase64 == null) toast("Pick a screenshot first") else audit()
        }
        findViewById<Button>(R.id.btnCopy).setOnClickListener { copyResult() }
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQ_PICK || resultCode != RESULT_OK || data?.data == null) return
        val uri = data.data!!
        runCatching {
            val bitmap = decodeScaled(uri)
            findViewById<ImageView>(R.id.imgPreview).setImageBitmap(bitmap)
            val bos = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, 85, bos)
            imageBase64 = Base64.encodeToString(bos.toByteArray(), Base64.NO_WRAP)
            setResultText("Screenshot loaded. Choose an action below.")
        }.onFailure { toast("Couldn't load that image") }
    }

    private fun decodeScaled(uri: Uri): Bitmap {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val src = ImageDecoder.createSource(contentResolver, uri)
            ImageDecoder.decodeBitmap(src) { decoder, info, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.isMutableRequired = true
                if (info.size.width > 1600) {
                    decoder.setTargetSize(1600, (info.size.height * 1600f / info.size.width).toInt())
                }
            }
        } else {
            @Suppress("DEPRECATION")
            BitmapFactory.decodeStream(contentResolver.openInputStream(uri))
        }
    }

    private fun analyze(spinnerTone: Spinner, task: LlmClient.AnalysisTask, base64: String) {
        val tone = Tone.entries[spinnerTone.selectedItemPosition]
        setResultText("Analyzing…")
        setBusy(true)
        LlmClient.analyzeImage(this, tone, task, base64) { result ->
            setBusy(false)
            result.fold(
                onSuccess = { (detected, replies) ->
                    val header = detected?.let { "Detected: $it\n\n" } ?: ""
                    setResultText(header + replies.joinToString("\n\n") { "• $it" })
                },
                onFailure = { e -> setResultText("Failed: ${e.message}") }
            )
        }
    }

    private fun audit() {
        setResultText("Auditing…")
        setBusy(true)
        LlmClient.auditImage(this, imageBase64!!) { result ->
            setBusy(false)
            result.fold(
                onSuccess = { setResultText(it) },
                onFailure = { e -> setResultText("Failed: ${e.message}") }
            )
        }
    }

    private fun setBusy(busy: Boolean) {
        listOf(R.id.btnOpeners, R.id.btnReplies, R.id.btnAudit).forEach {
            findViewById<Button>(it).isEnabled = !busy
        }
    }

    private fun setResultText(text: String) {
        lastResult = text
        findViewById<TextView>(R.id.tvResult).text = text
    }

    private fun copyResult() {
        if (lastResult.isEmpty() || "Analyzing…" == lastResult || "Auditing…" == lastResult ||
            lastResult.startsWith("Screenshot loaded") || lastResult.startsWith("Failed")
        ) {
            toast("Nothing to copy yet")
            return
        }
        val cm = getSystemService(CLIPBOARD_SERVICE) as android.content.ClipboardManager
        cm.setPrimaryClip(android.content.ClipData.newPlainText("AI Reply", lastResult))
        toast("Copied")
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()

    companion object {
        private const val REQ_PICK = 41
    }
}
