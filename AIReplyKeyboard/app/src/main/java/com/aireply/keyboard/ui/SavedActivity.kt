package com.aireply.keyboard.ui

import android.app.Activity
import android.app.AlertDialog
import android.content.ClipData
import android.content.ClipboardManager
import android.os.Bundle
import android.view.View
import android.widget.ArrayAdapter
import android.widget.ListView
import android.widget.TextView
import android.widget.Toast
import com.aireply.keyboard.R
import com.aireply.keyboard.ai.Prefs

class SavedActivity : Activity() {

    private lateinit var list: ListView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_saved)
        list = findViewById(R.id.listSaved)

        list.setOnItemClickListener { _, _, pos, _ ->
            val items = Prefs.savedReplies(this)
            if (pos !in items.indices) return@setOnItemClickListener
            val cm = getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
            cm.setPrimaryClip(ClipData.newPlainText("AI Reply", items[pos]))
            Toast.makeText(this, "Copied — paste it anywhere", Toast.LENGTH_SHORT).show()
        }

        list.setOnItemLongClickListener { _, _, pos, _ ->
            val items = Prefs.savedReplies(this)
            if (pos !in items.indices) return@setOnItemLongClickListener true
            AlertDialog.Builder(this)
                .setMessage("Delete this saved reply?")
                .setPositiveButton("Delete") { _, _ ->
                    Prefs.removeSaved(this, pos)
                    refresh()
                }
                .setNegativeButton("Cancel", null)
                .show()
            true
        }

        refresh()
    }

    private fun refresh() {
        val items = Prefs.savedReplies(this)
        list.adapter = ArrayAdapter(this, android.R.layout.simple_list_item_1, items)
        findViewById<TextView>(R.id.tvSavedEmpty).visibility =
            if (items.isEmpty()) View.VISIBLE else View.GONE
    }
}
