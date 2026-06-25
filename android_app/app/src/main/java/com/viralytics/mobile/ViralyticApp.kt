package com.viralytics.mobile

import android.app.Application
import com.ubtrobot.Robot

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
        Robot.initialize(this)
    }
}
