package com.aireply.keyboard.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class OfflineFallbackTest {

    @Test
    fun `always returns exactly three suggestions`() {
        Tone.entries.forEach { tone ->
            listOf("high_enthu", "low_dry", "angry_defensive", "medium_cool").forEach { mood ->
                assertEquals(3, OfflineFallback.suggest(tone, mood, hinglish = false).size)
                assertEquals(3, OfflineFallback.suggest(tone, mood, hinglish = true).size)
            }
        }
    }

    @Test
    fun `no empty suggestions`() {
        OfflineFallback.suggest(Tone.FLIRTY, "low_dry", false).forEach {
            assertTrue(it.isNotBlank())
        }
    }

    @Test
    fun `hinglish pool differs from english pool`() {
        val en = OfflineFallback.suggest(Tone.CASUAL, "medium_cool", false)
        val hi = OfflineFallback.suggest(Tone.CASUAL, "medium_cool", true)
        assertTrue(en != hi)
    }

    @Test
    fun `dry mood suggestions respect the dont-chase advice`() {
        val suggestions = OfflineFallback.suggest(Tone.CASUAL, "low_dry", false)
        assertTrue(suggestions.any { !it.contains("?") || it.contains("later") || it.contains("no rush") })
    }
}
