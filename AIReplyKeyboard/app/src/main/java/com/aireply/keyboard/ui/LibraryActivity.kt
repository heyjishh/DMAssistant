package com.aireply.keyboard.ui

import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.os.Bundle
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.ListView
import android.widget.Spinner
import android.widget.Toast
import com.aireply.keyboard.R
import com.aireply.keyboard.ai.LibraryData

class LibraryActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_library)

        val spinner = findViewById<Spinner>(R.id.spinnerCategory)
        val list = findViewById<ListView>(R.id.listLines)

        spinner.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item,
            LibraryData.categories.map { it.label }
        )
        spinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: android.view.View?, pos: Int, id: Long) {
                list.adapter = ArrayAdapter(
                    this@LibraryActivity,
                    android.R.layout.simple_list_item_1,
                    LibraryData.categories[pos].items
                )
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }

        list.setOnItemClickListener { _, _, pos, _ ->
            val line = LibraryData.categories[spinner.selectedItemPosition].items[pos]
            val cm = getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
            cm.setPrimaryClip(ClipData.newPlainText("AI Reply", line))
            Toast.makeText(this, "Copied — paste it in any chat", Toast.LENGTH_SHORT).show()
        }
    }
}
