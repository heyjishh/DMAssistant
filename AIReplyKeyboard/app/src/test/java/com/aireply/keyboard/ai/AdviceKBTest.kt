package com.aireply.keyboard.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AdviceKBTest {

    @Test
    fun `trap advice wins over mood advice`() {
        repeat(10) {
            val tip = AdviceKB.tip("high_enthu", "love_bombing")
            assertTrue(tip.contains("slow down") || tip.contains("intensity") || tip.contains("manifesting"))
        }
    }

    @Test
    fun `dry mood tips come from the dont-chase pool`() {
        repeat(20) {
            val tip = AdviceKB.tip("low_dry", null)
            assertTrue(
                "unexpected tip: [$tip]",
                tip.contains("chas") || tip.contains("Abundance") || tip.contains("exit") ||
                    tip.contains("24h") || tip.contains("needy")
            )
        }
    }

    @Test
    fun `every pool returns non-blank tips`() {
        listOf("high_enthu", "low_dry", "angry_defensive", "medium_cool").forEach { mood ->
            repeat(10) { assertTrue(AdviceKB.tip(mood, null).isNotBlank()) }
        }
    }

    @Test
    fun `pools have game knowledge coverage`() {
        // sanity: the KB contains the ported DMAssistant principles
        val all = AdviceKB::class.java.getDeclaredField("tips").apply { isAccessible = true }
            .get(null) as Map<*, *>
        assertTrue((all["opener"] as List<*>).any { it.toString().contains("20 words") })
        assertTrue((all["followup"] as List<*>).any { it.toString().contains("facts") || it.toString().contains("Feelings") || it.toString().contains("feelings") })
        assertEquals(9, all.size)
    }
}
