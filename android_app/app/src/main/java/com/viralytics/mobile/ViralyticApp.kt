package com.viralytics.mobile

import android.app.Application
import com.ubtrobot.Robot

class ViralyticApp : Application() {
    override fun onCreate() {
        super.onCreate()
        // Robot SDK must be initialized at Application level so all services
        // (MotionManager, SensorManager, NavigationManager, SpeechManager) are
        // fully bound before any Activity tries to get them via getSystemService().
        Robot.initialize(this)
    }
}
