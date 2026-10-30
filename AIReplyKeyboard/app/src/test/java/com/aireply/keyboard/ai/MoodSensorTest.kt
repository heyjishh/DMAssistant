package com.aireply.keyboard.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class MoodSensorTest {

    @Test
    fun `detects high enthusiasm`() {
        val read = MoodSensor.read("haha that's so funny\nomg yes 😂\nlol stop")
        assertEquals("high_enthu", read.mood)
        assertNull(read.trap)
    }

    @Test
    fun `detects low dry mood from short ok`() {
        val read = MoodSensor.read("so what are you up to\nnm\nk")
        assertEquals("low_dry", read.mood)
    }

    @Test
    fun `detects defensive mood`() {
        val read = MoodSensor.read("why did you say that\nseriously stop")
        assertEquals("angry_defensive", read.mood)
    }

    @Test
    fun `neutral conversation is medium cool`() {
        val read = MoodSensor.read("hey what did you do this weekend\nwent hiking, pretty tired now")
        assertEquals("medium_cool", read.mood)
    }

    @Test
    fun `detects love bombing trap early in conversation`() {
        val read = MoodSensor.read("i think you're perfect\nwe should get married")
        assertEquals("love_bombing", read.trap)
        assertNotNull(read.trapAdvice)
    }

    @Test
    fun `detects money trap`() {
        val read = MoodSensor.read("take me somewhere expensive\nbuy me a gift")
        assertEquals("money_talk", read.trap)
    }

    @Test
    fun `no trap in normal flirty chat`() {
        val read = MoodSensor.read("haha you're funny\nwhat are you doing this weekend")
        assertNull(read.trap)
    }

    @Test
    fun `advice exists for every mood`() {
        listOf("high_enthu", "low_dry", "angry_defensive", "medium_cool", "unknown").forEach {
            assertTrue(MoodSensor.moodAdvice(it).isNotEmpty())
        }
    }
}
