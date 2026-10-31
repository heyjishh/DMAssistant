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

    @Test
    fun `strips trailing period — reads more sincere in texts (Klin 2016)`() {
        val replies = LlmClient.parseReplies("sounds good.")
        assertEquals("sounds good", replies[0])
    }

    @Test
    fun `keeps question and exclamation endings`() {
        val replies = LlmClient.parseReplies("you coming? ||| hell yes!")
        assertEquals("you coming?", replies[0])
        assertEquals("hell yes!", replies[1])
    }

    @Test
    fun `dedupes identical starters by stripping leading filler`() {
        val replies = LlmClient.parseReplies("haha that's wild ||| haha same here ||| okay bet")
        assertTrue(replies[0].startsWith("haha"))
        assertTrue(!replies[1].startsWith("haha"))
        assertEquals("same here", replies[1])
    }

    @Test
    fun `keeps ellipsis intact`() {
        val replies = LlmClient.parseReplies("wait what… ||| no way")
        assertEquals("wait what…", replies[0])
    }
}
