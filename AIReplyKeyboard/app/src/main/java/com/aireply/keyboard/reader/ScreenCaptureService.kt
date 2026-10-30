package com.aireply.keyboard.reader

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.IBinder
import android.util.Base64
import java.io.ByteArrayOutputStream

/**
 * One-shot screen capture: started with the user's MediaProjection consent, grabs a
 * single frame, stores it in [ChatStore], and stops. Android 14 requires this to run
 * as a foreground service with type mediaProjection.
 */
class ScreenCaptureService : Service() {

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        android.util.Log.i("ScreenCapture", "service started")
        @Suppress("DEPRECATION")
        val data = intent?.getParcelableExtra<Intent>("data")
        // NOTE: the projection result code is the activity result — RESULT_OK is -1,
        // so it must NOT be validated as "positive".
        if (data == null) {
            stopSelf()
            return START_NOT_STICKY
        }
        val code = intent.getIntExtra("code", Int.MIN_VALUE)

        createChannel()
        val notification: Notification = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("AI Reply captured your screen")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }

        val pm = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        val projection: MediaProjection? = runCatching { pm.getMediaProjection(code, data) }.getOrNull()
        if (projection == null) {
            stopSelf()
            return START_NOT_STICKY
        }

        val bitmap = runCatching { captureFrame(projection) }
            .onFailure { android.util.Log.e("ScreenCapture", "capture failed", it) }
            .getOrNull()
        if (bitmap != null) {
            val bos = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, 80, bos)
            ChatStore.updateImage(Base64.encodeToString(bos.toByteArray(), Base64.NO_WRAP))
            android.util.Log.i("ScreenCapture", "captured ${bitmap.width}x${bitmap.height}")
        }
        projection.stop()
        stopSelf()
        return START_NOT_STICKY
    }

    private fun captureFrame(projection: MediaProjection): Bitmap {
        // Android 14 requires a registered callback before createVirtualDisplay().
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            projection.registerCallback(object : MediaProjection.Callback() {
                override fun onStop() {}
            }, android.os.Handler(android.os.Looper.getMainLooper()))
        }
        val metrics = resources.displayMetrics
        val width = metrics.widthPixels
        val height = metrics.heightPixels
        val reader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2)
        val display: VirtualDisplay = projection.createVirtualDisplay(
            "aireply-capture", width, height, metrics.densityDpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, reader.surface, null, null
        )

        var image = reader.acquireLatestImage()
        var tries = 0
        while (image == null && tries < 30) {
            Thread.sleep(100)
            tries++
            image = reader.acquireLatestImage()
        }
        checkNotNull(image) { "No frame captured" }

        val plane = image.planes[0]
        val buffer = plane.buffer
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val rowPadding = rowStride - pixelStride * width
        val raw = Bitmap.createBitmap(width + rowPadding / pixelStride, height, Bitmap.Config.ARGB_8888)
        raw.copyPixelsFromBuffer(buffer)
        image.close()
        display.release()
        reader.close()

        // Downscale for the vision API — max 1080px wide.
        return if (raw.width > 1080) {
            val scaled = Bitmap.createScaledBitmap(raw, 1080, height * 1080 / raw.width, true)
            scaled
        } else raw
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "Screen capture", NotificationManager.IMPORTANCE_LOW)
            )
        }
    }

    companion object {
        private const val CHANNEL_ID = "screen_capture"
        private const val NOTIFICATION_ID = 42
    }
}
