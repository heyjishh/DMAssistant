package com.aireply.keyboard.reader

/** Shared snapshot of the most recent on-screen text captured by the accessibility service. */
object ChatStore {
    @Volatile var latestText: String = ""
        private set

    @Volatile var packageName: String = ""
        private set

    @Volatile var updatedAt: Long = 0
        private set

    /** Base64 JPEG of the last live screen capture (📸 button), for vision-based suggestions. */
    @Volatile var latestImage: String? = null
        private set

    @Volatile var latestImageAt: Long = 0
        private set

    fun update(pkg: String, text: String) {
        latestText = text
        packageName = pkg
        updatedAt = System.currentTimeMillis()
    }

    fun updateImage(base64: String) {
        latestImage = base64
        latestImageAt = System.currentTimeMillis()
    }

    fun clear() {
        latestText = ""
        packageName = ""
        updatedAt = 0
        latestImage = null
        latestImageAt = 0
    }

    val isFresh: Boolean
        get() = latestText.isNotEmpty() && System.currentTimeMillis() - updatedAt < 60_000
}
