package com.aireply.keyboard.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ToneAndLibraryTest {

    @Test
    fun `all nine tones exist with labels and instructions`() {
        assertEquals(9, Tone.entries.size)
        Tone.entries.forEach {
            assertTrue(it.label.isNotBlank())
            assertTrue(it.instruction.length > 20)
        }
    }

    @Test
    fun `tone lookup by label works`() {
        assertEquals(Tone.SPICY, Tone.fromLabel("Spicy"))
        assertEquals(Tone.SMOOTH, Tone.fromLabel("Smooth"))
    }

    @Test
    fun `library has categories with content`() {
        assertTrue(LibraryData.categories.size >= 9)
        LibraryData.categories.forEach { category ->
            assertTrue(category.label.isNotBlank())
            assertTrue(category.items.isNotEmpty())
            category.items.forEach { assertTrue(it.isNotBlank()) }
        }
    }

    @Test
    fun `playbook provides research-backed sections`() {
        val cats = com.aireply.keyboard.ai.Playbook.asCategories()
        assertEquals(7, cats.size)
        assertTrue(cats.all { it.label.startsWith("📚") })
        assertTrue(cats.all { it.items.size >= 3 })
        assertTrue(cats.all { c -> c.items.all { it.contains("Why it works:") } })
        // spot-check the research claims are attributed
        val allText = cats.joinToString("") { it.items.joinToString("") }
        assertTrue(allText.contains("Dutton & Aron"))
        assertTrue(allText.contains("Aron et al. (1997)"))
        assertTrue(allText.contains("Klin (2016)"))
        assertTrue(allText.contains("Manson"))
        assertTrue(allText.contains("Attached"))
    }

    @Test
    fun `provider presets cover main options`() {
        assertEquals(5, ProviderPresets.all.size)
        val web2api = ProviderPresets.all.first { it.label.contains("web2api") }
        assertEquals("http://10.0.2.2:8081/v1", web2api.baseUrl)
        val custom = ProviderPresets.CUSTOM
        assertNull(custom.baseUrl)
    }

    @Test
    fun `moodsensor word matching avoids substring false positives`() {
        // "what" inside a normal question must not read as defensive
        assertEquals("medium_cool", MoodSensor.read("what are you up to tonight").mood)
        // genuine defensive phrase still detected
        assertEquals("angry_defensive", MoodSensor.read("stop texting me").mood)
    }
}
