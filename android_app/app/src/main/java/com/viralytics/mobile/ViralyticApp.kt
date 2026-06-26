package com.viralytics.mobile

import android.app.Application
import coil.Coil
import coil.ImageLoader
import com.ubtrobot.Robot
import okhttp3.OkHttpClient

/**
 * Custom Application class — entry point for the entire app process.
 *
 * WHY THIS EXISTS:
 * The UBTech Cruzr SDK v2.8.0 requires Robot.initialize() to be called with an
 * Application context before ANY Activity is created. Calling it in Activity.onCreate()
 * causes gesture failures (status:5 FAILED - jointmotion) because the SDK's internal
 * service bindings are not yet established when the Activity tries to use them.
 *
 * Registered in AndroidManifest.xml via android:name=".ViralyticApp" on <application>.
 */
class ViralyticApp : Application() {
    override fun onCreate() {
        super.onCreate()
        try {
            Robot.initialize(this)
        } catch (_: Throwable) {
            // Not a CRUZR robot — SDK not available at runtime on regular phones
        }
        setupCoil()
    }

    private fun setupCoil() {
        val trustAll = arrayOf<javax.net.ssl.TrustManager>(object : javax.net.ssl.X509TrustManager {
            override fun checkClientTrusted(chain: Array<java.security.cert.X509Certificate>, authType: String) {}
            override fun checkServerTrusted(chain: Array<java.security.cert.X509Certificate>, authType: String) {}
            override fun getAcceptedIssuers(): Array<java.security.cert.X509Certificate> = arrayOf()
        })
        val sslCtx = javax.net.ssl.SSLContext.getInstance("TLS")
        sslCtx.init(null, trustAll, java.security.SecureRandom())
        val coilClient = OkHttpClient.Builder()
            .sslSocketFactory(sslCtx.socketFactory, trustAll[0] as javax.net.ssl.X509TrustManager)
            .hostnameVerifier { _, _ -> true }
            .build()
        @Suppress("DEPRECATION")
        Coil.setImageLoader(ImageLoader.Builder(this).okHttpClient(coilClient).build())
    }
}
