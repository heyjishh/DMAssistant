package com.aireply.keyboard.reader

/** Shared snapshot of the most recent on-screen text captured by the accessibility service. */
object ChatStore {
    @Volatile var latestText: String = ""
        private set

    @Volatile var packageName: String = ""
        private set

    @Volatile var updatedAt: Long = 0
        private set

    fun update(pkg: String, text: String) {
        latestText = text
        packageName = pkg
        updatedAt = System.currentTimeMillis()
    }

    fun clear() {
        latestText = ""
        packageName = ""
        updatedAt = 0
    }

    val isFresh: Boolean
        get() = latestText.isNotEmpty() && System.currentTimeMillis() - updatedAt < 60_000
}
