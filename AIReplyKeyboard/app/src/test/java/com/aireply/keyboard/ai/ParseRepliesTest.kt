package com.aireply.keyboard.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ParseRepliesTest {

    @Test
    fun `parses pipe-delimited replies`() {
        val replies = LlmClient.parseReplies("Hey there! ||| What's up? ||| Talk later?")
        assertEquals(3, replies.size)
        assertEquals("Hey there!", replies[0])
    }

    @Test
    fun `parses newline-delimited replies when model ignores the delimiter`() {
        val replies = LlmClient.parseReplies("Sounds good, let's catch up soon!\nYeah, I'm down for whatever.\nNo worries at all!")
        assertEquals(3, replies.size)
        assertEquals("Sounds good, let's catch up soon!", replies[0])
    }

    @Test
    fun `strips list numbering and bullets`() {
        val replies = LlmClient.parseReplies("1. First one\n- Second one\n• Third one")
        assertEquals(listOf("First one", "Second one", "Third one"), replies)
    }

    @Test
    fun `caps at three and drops empties`() {
        val replies = LlmClient.parseReplies("a ||| ||| b ||| c ||| d")
        assertEquals(3, replies.size)
    }

    @Test
    fun `throws on no usable content`() {
        var threw = false
        try {
            LlmClient.parseReplies("   ")
        } catch (e: Exception) {
            threw = e.message?.contains("no usable") == true
        }
        assertTrue(threw)
    }
}
