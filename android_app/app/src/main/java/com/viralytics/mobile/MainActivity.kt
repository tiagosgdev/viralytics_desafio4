package com.viralytics.mobile

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import java.io.File
import android.graphics.Typeface
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.util.Base64
import android.util.TypedValue
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.chip.Chip
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.view.ViewGroup
import android.widget.ImageView
import coil.load
import coil.transform.RoundedCornersTransformation
import com.google.android.material.textfield.TextInputLayout
import com.viralytics.mobile.databinding.ActivityMainBinding
import org.json.JSONArray
import org.json.JSONObject

// V2.8.0 UBTech Robot Imports
import com.ubtrobot.Robot
import com.ubtrobot.navigation.NavigationManager
import com.ubtrobot.navigation.NavigationOption
import com.ubtrobot.navigation.Location
import com.ubtrobot.speech.SpeechManager
import com.ubtrobot.navigation.Point
import android.net.Uri
import com.ubtrobot.motion.MotionManager
import com.ubtrobot.motion.PerformingOption
import com.ubtrobot.sensor.SensorManager as CruzrSensorManager
import com.ubtrobot.sensor.SensorListener
import com.ubtrobot.sensor.SensorEvent
import com.ubtrobot.sensor.SensorDevice

import org.eclipse.paho.client.mqttv3.MqttClient
import org.eclipse.paho.client.mqttv3.MqttConnectOptions
import org.eclipse.paho.client.mqttv3.MqttMessage
import org.eclipse.paho.client.mqttv3.MqttCallbackExtended
import org.eclipse.paho.client.mqttv3.IMqttDeliveryToken
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence

/**
 * Main activity for the Viralytics tablet app running on the Cruzr robot.
 *
 * Architecture overview:
 *   PC Server (FastAPI)  ──MQTT──►  This app  ──UBTech SDK──►  Robot hardware
 *
 * Communication:
 *   - Subscribes to MQTT topic "cruzr/commands" to receive navigation/speech/gesture commands
 *   - Publishes to MQTT topic "cruzr/status" to report navigation events back to the server
 *   - MQTT broker: test.mosquitto.org:8883 (TLS encrypted, no auth — demo only)
 *   - HTTP: calls the FastAPI server directly for outfit scanning and chat
 *
 * Robot control (UBTech Cruzr SDK v2.8.0):
 *   - NavigationManager: navigates to map markers or raw (x, y, theta) coordinates
 *   - MotionManager:     plays gesture animations (raise arms, wave, goodbye…)
 *   - SensorManager:     reads the LIDAR human-detection sensor
 *   - SpeechManager/TTS: speaks text using the robot's onboard TTS engine
 *
 * Customer session lifecycle:
 *   door sensor → greet command → navigate to entrance → LIDAR detects person
 *   → speak + gesture → session starts (3-min idle timer) → customer scans outfit
 *   → each scan/chat resets the timer → AI sends guide_user → navigate to stand
 *   → arrive → speak → session ends → robot returns to entrance
 */
enum class AppMode { TABLET, PHONE_CAMERA }


class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val viewModel: MainViewModel by viewModels()

    private var appMode: AppMode = AppMode.PHONE_CAMERA

    // --- NAVIGATION CONFIG ---
    private val TRACK_MODE = false       // false = point-to-point; true = follow path polyline
    private val NAV_MAX_SPEED = 0.5f    // metres/second
    private val NAV_RETRY_COUNT = 2
    private val NAV_RETRY_INTERVAL = 2000

    // Loaded from assets/coordinates.json at startup: stand name → (x, y, theta)
    private var standCoordinates: Map<String, Triple<Float, Float, Float>> = emptyMap()

    // Maps vision-model category codes → actual marker names on the robot's nav map.
    // Marker names must match coordinates.json / the labels placed in the UBTECH dashboard.
    private val CATEGORY_STAND_MAP = mapOf(
        // Tops
        "short_sleeve_top"    to "T-shirt Stand",
        "long_sleeve_top"     to "T-shirt Stand",
        "vest"                to "T-shirt Stand",
        "top"                 to "T-shirt Stand",
        "tops"                to "T-shirt Stand",
        // Outerwear
        "long_sleeve_outwear" to "Jacket Stand",
        "outwear"             to "Jacket Stand",
        "jacket"              to "Jacket Stand",
        "coat"                to "Jacket Stand",
        "hoodie"              to "Jacket Stand",
        "sweater"             to "Jacket Stand",
        // Bottoms
        "trousers"            to "Pants Stand",
        "shorts"              to "Pants Stand",
        "jeans"               to "Pants Stand",
        "bottom"              to "Pants Stand",
        "bottoms"             to "Pants Stand",
        // Dresses / Skirts
        "skirt"               to "Dress Stand",
        "short_sleeve_dress"  to "Dress Stand",
        "long_sleeve_dress"   to "Dress Stand",
        "vest_dress"          to "Dress Stand",
        "sling_dress"         to "Dress Stand",
        "dress"               to "Dress Stand",
        "dresses"             to "Dress Stand",
    )

    private var currentTab: String = "scan"

    // MQTT client — connects to test.mosquitto.org on a background thread
    private var mqttClient: MqttClient? = null

    // Fallback reconnect: if paho's isAutomaticReconnect silently fails (known Android/TLS issue),
    // this fires 15 s after connectionLost and fully restarts the MQTT stack.
    private val mqttReconnectHandler = Handler(Looper.getMainLooper())
    private val mqttReconnectRunnable = Runnable {
        if (mqttClient?.isConnected == true) return@Runnable
        Thread {
            try { mqttClient?.close() } catch (_: Exception) {}
            mqttClient = null
            startMqttListener()
        }.start()
    }

    // UBTech SDK hardware managers — obtained from Robot singleton after Robot.initialize()
    private var navigationManager: NavigationManager? = null
    private var speechManager: SpeechManager? = null
    private var motionManager: MotionManager? = null
    private var cruzrSensorManager: CruzrSensorManager? = null
    private lateinit var textToSpeech: TextToSpeech

    // isBusy: true while any navigation is in progress — blocks stacking navigation commands.
    // isAtEntrance: true after arriving at the entrance position, while waiting for LIDAR detection.
    // Both are @Volatile because the MQTT background thread reads them while the main thread writes.
    @Volatile private var isBusy = false
    @Volatile private var isAtEntrance = false
    private var humanDetectListener: SensorListener? = null
    @Volatile private var pendingGreetText  = "Bem-vindo! Sou o Cruzr, o seu assistente de moda pessoal. Aproxime-se e deixe-me ajudá-lo a encontrar o outfit perfeito!"
    @Volatile private var pendingGreetGesture = "raise"

    // Customer session — blocks new greet commands while serving; returns robot to entrance when done
    @Volatile private var isServingCustomer = false
    private var entranceX = 0f
    private var entranceY = 0f
    private var entranceTheta = 0f
    private var entranceCoordsSet = false
    private val sessionHandler = Handler(Looper.getMainLooper())
    private val sessionTimeoutRunnable = Runnable { endCustomerSession("timeout") }
    private val SESSION_TIMEOUT_MS  = 3 * 60 * 1000L   // 3 minutes
    private val LIDAR_WAIT_TIMEOUT_MS = 25 * 1000L      // 25 seconds — nobody showed up at entrance
    private val lidarWaitHandler = Handler(Looper.getMainLooper())
    private val lidarWaitTimeoutRunnable = Runnable {
        if (isAtEntrance) {
            android.util.Log.i("CruzrApp", "LIDAR wait timeout — nobody arrived, releasing busy lock")
            isAtEntrance = false
            stopHumanDetection()
            isBusy = false
            publishStatus("lidar_timeout", null)
            runOnUiThread { setStatus("Nobody arrived — ready for next customer") }
        }
    }

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                openCameraActivity()
            } else {
                setStatus("Camera permission denied.")
            }
        }

    private val cameraActivityLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode != RESULT_OK) {
                setStatus("Capture cancelled.")
                return@registerForActivityResult
            }
            val path = result.data?.getStringExtra(CameraActivity.RESULT_IMAGE_PATH) ?: run {
                setStatus("Capture failed.")
                return@registerForActivityResult
            }
            val bitmap = BitmapFactory.decodeFile(path)?.applyExifRotation(path)
            File(path).delete()
            if (bitmap == null) {
                setStatus("Failed to decode captured image.")
                return@registerForActivityResult
            }
            val baseUrl = normalizedBaseUrl() ?: return@registerForActivityResult
            viewModel.uploadScan(bitmap, baseUrl, loadPersona())
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        appMode = detectAppMode()
        setupUiForMode(appMode)

        standCoordinates = loadStandCoordinates()

        if (appMode == AppMode.TABLET) {
            initCruzrHardware()
        }
        startMqttListener()

        observeViewModel()

        setStatus(if (appMode == AppMode.TABLET) "Waiting for scan from phone…" else "Ready to scan.")
        updateSessionLabel()
        renderDetections()
        renderRecommendations()
        switchTab("scan")

        textToSpeech = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech.language = java.util.Locale.US
                android.util.Log.d("CruzrApp", "TTS Ready!")
            }
        }

        binding.captureButton.setOnClickListener {
            if (loadPersona().isBlank()) { showPersonaDialog(); return@setOnClickListener }
            launchCamera()
        }
        binding.sendChatButton.setOnClickListener { sendChat() }
        binding.chatInput.setOnEditorActionListener { _, _, _ -> sendChat(); true }
        binding.recommendationsLeftButton.setOnClickListener { scrollRecommendations(-1) }
        binding.recommendationsRightButton.setOnClickListener { scrollRecommendations(1) }
        binding.connectionSettingsButton.setOnClickListener { showConnectionSettingsDialog() }
        binding.switchPersonaButton?.setOnClickListener { showPersonaDialog() }
        binding.tabScanButton.setOnClickListener { switchTab("scan") }
        binding.tabRefineButton.setOnClickListener { switchTab("refine") }

        val storedPersona = loadPersona()
        if (storedPersona.isNotBlank()) {
            applyPersona(storedPersona)
        } else {
            showPersonaDialog()
        }
    }

    private fun observeViewModel() {
        viewModel.events.observe(this) { event ->
            when (event) {
                is UiEvent.SetStatus -> setStatus(event.message)
                is UiEvent.ShowToast -> toast(event.message)
                is UiEvent.ScanComplete -> {
                    if (appMode == AppMode.PHONE_CAMERA) {
                        toast("Scan sent to tablet!")
                    } else {
                        setStatus("Scan received from phone.")
                        renderDetections()
                        renderRecommendations()
                        updateAnnotatedImage(event.annotatedFrameBase64)
                        switchTab("scan")
                        updateSessionLabel("Vision-led")
                        showChatReply("Scan complete. Tap a recommendation to inspect it, or refine with chat.")
                        extendSession()
                    }
                }
                is UiEvent.ScanError -> showChatReply(event.message)
                is UiEvent.ChatComplete -> {
                    renderRecommendations()
                    switchTab("refine")
                    val mode = if (binding.replaceVisionSwitch.isChecked) "Search-led override" else "Vision + search"
                    updateSessionLabel(mode)
                    showChatReply(event.reply)
                    binding.chatInput.text?.clear()
                    extendSession()
                }
                is UiEvent.AgentRecsComplete -> {
                    if (appMode == AppMode.TABLET) renderRecommendations()
                }
                is UiEvent.ChatError -> showChatReply(event.message)
            }
        }
    }

    private fun detectAppMode(): AppMode {
        val override = getSharedPreferences("viralytics_mobile", Context.MODE_PRIVATE)
            .getString("device_mode_override", null)
        if (override == "tablet") return AppMode.TABLET
        if (override == "phone") return AppMode.PHONE_CAMERA

        return try {
            if (Robot.globalContext() != null) AppMode.TABLET else AppMode.PHONE_CAMERA
        } catch (_: Throwable) {
            AppMode.PHONE_CAMERA
        }
    }

    private fun saveDeviceModeOverride(mode: AppMode) {
        getSharedPreferences("viralytics_mobile", Context.MODE_PRIVATE)
            .edit()
            .putString("device_mode_override", if (mode == AppMode.TABLET) "tablet" else "phone")
            .apply()
    }

    private fun setupUiForMode(mode: AppMode) {
        when (mode) {
            AppMode.PHONE_CAMERA -> {
                requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
                binding.tabBar?.isVisible = false
                binding.refineSection.isVisible = false
                binding.modeIndicatorText?.text = "CAMERA MODE"
            }
            AppMode.TABLET -> {
                requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE
                binding.captureButton.isVisible = false
                binding.scanWaitingText?.isVisible = true
                binding.modeIndicatorText?.text = "DISPLAY MODE"
            }
        }
    }

    private fun launchCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            return
        }
        openCameraActivity()
    }

    private fun openCameraActivity() {
        cameraActivityLauncher.launch(Intent(this, CameraActivity::class.java))
    }

    private fun sendChat() {
        val persona = loadPersona()
        if (persona.isBlank()) {
            showPersonaDialog()
            return
        }
        val message = binding.chatInput.text.toString().trim()
        if (message.isBlank()) {
            toast("Enter a refinement message first.")
            return
        }
        val baseUrl = normalizedBaseUrl() ?: return
        viewModel.sendChat(
            message = message,
            baseUrl = baseUrl,
            replaceVision = binding.replaceVisionSwitch.isChecked,
            persona = persona,
        )
    }

    private fun renderDetections() {
        binding.detectionsGroup.removeAllViews()
        val categories = viewModel.detectedCategories
        if (categories.isEmpty()) {
            val chip = buildDetectionChip(getString(R.string.detections_empty))
            binding.detectionsGroup.addView(chip)
            return
        }
        categories.distinct().forEach { category ->
            binding.detectionsGroup.addView(buildDetectionChip(category.replace("_", " ")))
        }
    }

    private fun buildDetectionChip(label: String): Chip {
        val isCruella = viewModel.selectedPersona == "cruella"
        return Chip(this).apply {
            text = label.replaceFirstChar { it.uppercase() }
            isClickable = false
            isCheckable = false
            chipBackgroundColor = ContextCompat.getColorStateList(context,
                if (isCruella) R.color.cruella_surface_soft else R.color.brand_surface_soft)
            chipStrokeColor = ContextCompat.getColorStateList(context,
                if (isCruella) R.color.cruella_border else R.color.brand_border)
            chipStrokeWidth = dp(1f)
            setTextColor(ContextCompat.getColor(context,
                if (isCruella) R.color.cruella_text else R.color.brand_text))
        }
    }

    private fun renderRecommendations() {
        val recommendations = viewModel.currentRecommendations
        binding.recommendationsStrip.removeAllViews()
        val hasItems = recommendations.isNotEmpty()
        binding.recommendationsEmptyText.isVisible = !hasItems
        binding.recommendationsScroll.isVisible = hasItems
        binding.recommendationsLeftButton.isEnabled = hasItems
        binding.recommendationsRightButton.isEnabled = hasItems

        if (!hasItems) return

        recommendations.forEachIndexed { index, item ->
            binding.recommendationsStrip.addView(buildRecommendationCard(item, index))
        }
        binding.recommendationsScroll.post {
            binding.recommendationsScroll.scrollTo(0, 0)
        }
    }

    private fun buildRecommendationCard(item: RecommendationItem, index: Int): View {
        val isCruella = viewModel.selectedPersona == "cruella"
        val surfaceColor = ContextCompat.getColor(this, if (isCruella) R.color.cruella_surface else R.color.brand_surface)
        val surfaceSoftColor = ContextCompat.getColor(this, if (isCruella) R.color.cruella_surface_soft else R.color.brand_surface_soft)
        val textColor = ContextCompat.getColor(this, if (isCruella) R.color.cruella_text else R.color.brand_text)
        val mutedColor = ContextCompat.getColor(this, if (isCruella) R.color.cruella_muted else R.color.brand_muted)
        val accentColor = ContextCompat.getColor(this, if (isCruella) R.color.cruella_accent_strong else R.color.brand_accent_strong)
        val borderColor = ContextCompat.getColor(this, if (isCruella) R.color.cruella_border else R.color.brand_border)


        val cardWidth = resources.getDimensionPixelSize(R.dimen.rec_card_width)
        val card = MaterialCardView(this).apply {
            layoutParams = LinearLayout.LayoutParams(cardWidth, LinearLayout.LayoutParams.MATCH_PARENT).also {
                it.marginEnd = dp(12)
            }
            radius = dp(22).toFloat()
            strokeWidth = dp(1)
            setStrokeColor(borderColor)
            cardElevation = 0f
            setCardBackgroundColor(surfaceColor)
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
                foreground = ContextCompat.getDrawable(context, android.R.drawable.list_selector_background)
            }
            isClickable = true
            isFocusable = true
            setOnClickListener { showRecommendationDetail(item) }
        }

        val root = LinearLayout(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT
            )
            orientation = LinearLayout.VERTICAL
        }

        // Image area
        val recImageHeight = resources.getDimensionPixelSize(R.dimen.rec_card_image_height)
        val imageView = ImageView(this).apply {
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, recImageHeight)
            scaleType = ImageView.ScaleType.CENTER_CROP
            setBackgroundColor(surfaceSoftColor)
        }
        val proxyUrl = imageProxyUrl(item.imageUrl)
        if (proxyUrl != null) {
            imageView.load(proxyUrl) {
                crossfade(true)
                error(android.R.drawable.ic_menu_gallery)
            }
        }
        root.addView(imageView)

        // Text content
        val content = LinearLayout(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(10), dp(12), dp(12))
        }

        content.addView(TextView(this).apply {
            text = item.category.replace("_", " ").uppercase()
            setTextColor(mutedColor)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 9f)
            setTypeface(typeface, Typeface.BOLD)
            letterSpacing = 0.08f
        })

        content.addView(TextView(this).apply {
            text = item.name
            setTextColor(textColor)
            setTypeface(typeface, Typeface.BOLD)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
            maxLines = 2
            ellipsize = android.text.TextUtils.TruncateAt.END
            setPadding(0, dp(3), 0, 0)
        })

        if (!item.description.isNullOrBlank()) {
            content.addView(TextView(this).apply {
                text = item.description
                setTextColor(mutedColor)
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
                maxLines = 2
                ellipsize = android.text.TextUtils.TruncateAt.END
                setPadding(0, dp(4), 0, 0)
            })
        }

        content.addView(TextView(this).apply {
            text = item.price
            setTextColor(accentColor)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            setTypeface(typeface, Typeface.BOLD)
            setPadding(0, dp(6), 0, 0)
        })

        root.addView(content)

        if (index == viewModel.currentRecommendations.lastIndex) {
            (card.layoutParams as LinearLayout.LayoutParams).marginEnd = 0
        }

        card.addView(root)
        return card
    }

    private fun imageProxyUrl(imageUrl: String?): String? {
        if (imageUrl.isNullOrBlank()) return null
        val base = loadServerUrl().trimEnd('/')
        return "$base/api/image-proxy?url=${java.net.URLEncoder.encode(imageUrl, "UTF-8")}"
    }

    private fun scrollRecommendations(direction: Int) {
        val child = binding.recommendationsStrip.getChildAt(0) ?: return
        val step = child.width + dp(12)
        val scroll = binding.recommendationsScroll
        val maxScroll = (binding.recommendationsStrip.width - scroll.width).coerceAtLeast(0)
        val current = scroll.scrollX
        val target = when {
            direction > 0 && current >= maxScroll - dp(8) -> 0
            direction < 0 && current <= dp(8) -> maxScroll
            else -> (current + direction * step).coerceIn(0, maxScroll)
        }
        scroll.smoothScrollTo(target, 0)
    }

    private fun showRecommendationDetail(item: RecommendationItem) {
        val dialogView = LayoutInflater.from(this).inflate(R.layout.dialog_recommendation_detail, null)

        val detailImage = dialogView.findViewById<ImageView>(R.id.detailImage)
        val proxyUrl = imageProxyUrl(item.imageUrl)
        if (proxyUrl != null) {
            detailImage.visibility = View.VISIBLE
            detailImage.load(proxyUrl) {
                crossfade(true)
                error(android.R.drawable.ic_menu_gallery)
            }
        }

        dialogView.findViewById<TextView>(R.id.detailName).text = item.name
        dialogView.findViewById<TextView>(R.id.detailCategory).text = item.category.replace("_", " ").uppercase()
        dialogView.findViewById<TextView>(R.id.detailPrice).text = item.price
        dialogView.findViewById<TextView>(R.id.detailReason).text =
            item.reason.ifBlank { "Recommended from your search context." }
        dialogView.findViewById<TextView>(R.id.detailDescription).text =
            item.description?.takeIf { it.isNotBlank() } ?: getString(R.string.no_description)

        val storeRows = dialogView.findViewById<LinearLayout>(R.id.detailStoreRows)
        val attributeRows = dialogView.findViewById<LinearLayout>(R.id.detailAttributeRows)
        storeRows.removeAllViews()
        attributeRows.removeAllViews()

        addDetailRow(storeRows, "type", item.category.replace("_", " "), "type", item)
        item.brand?.let { addDetailRow(storeRows, "brand", it, "brand", item) }
        item.sku?.let { addDetailRow(storeRows, "sku", it, "sku", item) }
        item.stockStatus?.let { addDetailRow(storeRows, "stock", it.replace("_", " "), "stock_status", item) }
        if (item.sizes.isNotEmpty()) {
            addDetailRow(storeRows, "sizes", item.sizes.joinToString(", "), "sizes", item)
        }

        if (item.metadata.isEmpty()) {
            addPlaceholderRow(attributeRows, getString(R.string.no_attribute_rows))
        } else {
            item.metadata.forEach { (key, value) ->
                addDetailRow(attributeRows, key.replace("_", " "), value, key, item)
            }
        }

        setupFeedbackSection(dialogView, item)

        MaterialAlertDialogBuilder(this)
            .setView(dialogView)
            .setPositiveButton("Take me there!") { _, _ ->
                val baseUrl = normalizedBaseUrl()
                if (baseUrl == null) { setStatus("Server URL not configured."); return@setPositiveButton }
                speakText(if (item.reason.isNotBlank())
                    "Encontrei uma ótima opção! ${item.reason}. Siga-me, vou mostrar-lhe onde está."
                else
                    "Siga-me, vou mostrar-lhe onde fica o ${item.category.replace('_', ' ')}.")
                publishStatus("navigation_started", JSONObject().put("category", item.category))
                viewModel.navigateByCategory(baseUrl, item.category)
            }
            .setNegativeButton("Close", null)
            .show()
    }

    private fun setupFeedbackSection(dialogView: View, item: RecommendationItem) {
        val section = dialogView.findViewById<LinearLayout>(R.id.detailFeedbackSection)
        val row = dialogView.findViewById<LinearLayout>(R.id.detailFeedbackRow)
        val status = dialogView.findViewById<TextView>(R.id.detailFeedbackStatus)

        val baseUrl = normalizedBaseUrl()
        val canRate = viewModel.currentRoundId != null && item.itemId != null && baseUrl != null
        section.isVisible = canRate
        if (!canRate) return

        status.text = "How happy are you with this pick?"
        row.removeAllViews()
        val emojis = listOf("😣", "😕", "😐", "🙂", "😍")
        emojis.forEachIndexed { index, emoji ->
            val rating = index + 1
            val view = TextView(this).apply {
                text = emoji
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 28f)
                setPadding(dp(6), dp(2), dp(6), dp(2))
                isClickable = true
                setOnClickListener {
                    status.text = "Sending…"
                    viewModel.submitFeedback(baseUrl, item, rating) { msg -> status.text = msg }
                }
            }
            row.addView(view)
        }
    }

    private fun addPlaceholderRow(container: LinearLayout, text: String) {
        container.addView(TextView(this).apply {
            this.text = text
            setTextColor(ContextCompat.getColor(context, R.color.brand_muted))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
        })
    }

    private fun addDetailRow(
        container: LinearLayout,
        label: String,
        value: String,
        field: String,
        recommendation: RecommendationItem
    ) {
        val matches = attributeMatchesUserIntent(field, value, recommendation)
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(12), dp(14), dp(12))
            background = ContextCompat.getDrawable(
                context,
                if (matches) R.drawable.mobile_detail_row_match else R.drawable.mobile_detail_row
            )
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).also { it.bottomMargin = dp(10) }
        }

        row.addView(LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(TextView(context).apply {
                text = label.uppercase()
                setTextColor(ContextCompat.getColor(context, if (matches) R.color.brand_accent_strong else R.color.brand_muted))
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
                setTypeface(typeface, Typeface.BOLD)
            })
            if (matches) {
                addView(TextView(context).apply {
                    text = getString(R.string.match_badge)
                    setTextColor(ContextCompat.getColor(context, R.color.brand_accent_strong))
                    setTextSize(TypedValue.COMPLEX_UNIT_SP, 10f)
                    setTypeface(typeface, Typeface.BOLD)
                    background = ContextCompat.getDrawable(context, R.drawable.mobile_empty_state)
                    setPadding(dp(8), dp(3), dp(8), dp(3))
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.WRAP_CONTENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    ).also { it.marginStart = dp(8) }
                })
            }
        })

        row.addView(TextView(this).apply {
            text = value
            setTextColor(ContextCompat.getColor(context, if (matches) R.color.brand_accent_strong else R.color.brand_text))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
            setPadding(0, dp(6), 0, 0)
        })

        container.addView(row)
    }

    private fun getActiveIncludeFilters(): JSONObject? {
        val direct = extractIncludeFilters(viewModel.currentIncludeFilters)
        if (direct != null && direct.length() > 0) return direct

        val stateFilters = viewModel.currentConversationState?.optJSONObject("filters")
        val fallback = extractIncludeFilters(stateFilters)
        return if (fallback != null && fallback.length() > 0) fallback else null
    }

    private fun extractIncludeFilters(source: JSONObject?): JSONObject? {
        if (source == null) return null
        if (source.has("include")) {
            return source.optJSONObject("include")
        }
        val keys = source.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            if (source.optJSONArray(key) != null) {
                return source
            }
        }
        return null
    }

    private fun attributeMatchesUserIntent(field: String, value: String, recommendation: RecommendationItem): Boolean {
        val include = getActiveIncludeFilters() ?: return false
        val desired = include.optJSONArray(field) ?: return false
        val desiredValues = (0 until desired.length()).mapNotNull { index ->
            desired.optString(index)?.trim()?.lowercase()?.takeIf { it.isNotBlank() }
        }
        if (desiredValues.isEmpty()) return false
        return if (field == "type") {
            desiredValues.contains(recommendation.category.trim().lowercase())
        } else {
            desiredValues.contains(value.trim().lowercase())
        }
    }

    private fun updateAnnotatedImage(base64Image: String?) {
        if (base64Image.isNullOrBlank()) return
        try {
            val bytes = Base64.decode(base64Image, Base64.DEFAULT)
            binding.resultImage.setImageBitmap(BitmapFactory.decodeByteArray(bytes, 0, bytes.size))
        } catch (_: IllegalArgumentException) {
            binding.resultImage.setImageBitmap(null)
        }
    }

    private fun normalizedBaseUrl(): String? {
        val raw = loadServerUrl().trim().removeSuffix("/")
        if (raw.isBlank()) {
            toast("Enter the PC server URL first.")
            return null
        }
        return raw
    }

    private fun setStatus(message: String) {
        binding.statusText.text = message
        val dotColor = when {
            message.contains("error", ignoreCase = true) ||
            message.contains("fail", ignoreCase = true) ||
            message.contains("denied", ignoreCase = true) ||
            message.contains("lost", ignoreCase = true) -> R.color.brand_red
            message.contains("ready", ignoreCase = true) ||
            message.contains("complete", ignoreCase = true) ||
            message.contains("connected", ignoreCase = true) ||
            message.contains("saved", ignoreCase = true) ||
            message.contains("active", ignoreCase = true) -> R.color.brand_green
            else -> R.color.brand_accent
        }
        binding.statusDot?.backgroundTintList = ContextCompat.getColorStateList(this, dotColor)
    }

    private fun updateSessionLabel(mode: String? = null) {
        val sessionId = viewModel.currentSessionId
        binding.sessionText.text = if (sessionId.isNullOrBlank()) {
            getString(R.string.session_waiting)
        } else {
            val modeSuffix = if (mode.isNullOrBlank()) "" else " | $mode"
            "Session: ${sessionId.take(8)}$modeSuffix"
        }
    }

    private fun toast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    private fun loadServerUrl(): String {
        return getSharedPreferences("viralytics_mobile", Context.MODE_PRIVATE)
            .getString("server_url", "http://192.168.1.80:8000")
            .orEmpty()
    }

    private fun saveServerUrl(url: String) {
        getSharedPreferences("viralytics_mobile", Context.MODE_PRIVATE)
            .edit()
            .putString("server_url", url.trim())
            .apply()
    }

    private fun showConnectionSettingsDialog() {
        val dialogView = LayoutInflater.from(this).inflate(R.layout.dialog_connection_settings, null)
        val input = dialogView.findViewById<com.google.android.material.textfield.TextInputEditText>(R.id.dialogServerUrlInput)
        input.setText(loadServerUrl())

        val modeGroup = dialogView.findViewById<android.widget.RadioGroup>(R.id.deviceModeGroup)
        val phoneRadio = dialogView.findViewById<android.widget.RadioButton>(R.id.modePhoneRadio)
        val tabletRadio = dialogView.findViewById<android.widget.RadioButton>(R.id.modeTabletRadio)
        if (appMode == AppMode.TABLET) tabletRadio.isChecked = true else phoneRadio.isChecked = true

        MaterialAlertDialogBuilder(this)
            .setTitle(getString(R.string.connection_title))
            .setView(dialogView)
            .setPositiveButton(getString(R.string.connection_save)) { _, _ ->
                saveServerUrl(input.text?.toString().orEmpty())
                val selectedMode = if (tabletRadio.isChecked) AppMode.TABLET else AppMode.PHONE_CAMERA
                if (selectedMode != appMode) {
                    saveDeviceModeOverride(selectedMode)
                    recreate()
                } else {
                    toast("Connection saved.")
                }
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun switchTab(tab: String) {
        currentTab = tab
        val showingScan = tab == "scan"
        binding.scanSection.isVisible = showingScan
        binding.refineSection.isVisible = !showingScan
        styleTabButton(binding.tabScanButton, selected = showingScan)
        styleTabButton(binding.tabRefineButton, selected = !showingScan)
    }

    private fun showChatReply(message: String) {
        binding.chatReplyText.text = message
        binding.chatReplyText.isVisible = true
    }

    private fun styleTabButton(button: MaterialButton, selected: Boolean) {
        val isCruella   = viewModel.selectedPersona == "cruella"
        val activeBg    = ContextCompat.getColor(this, if (isCruella) R.color.cruella_accent       else R.color.brand_text)
        val inactiveBg  = ContextCompat.getColor(this, if (isCruella) R.color.cruella_surface_soft else R.color.brand_surface_soft)
        val inactiveText = ContextCompat.getColor(this, if (isCruella) R.color.cruella_text        else R.color.brand_text)
        val borderCol   = ContextCompat.getColor(this, if (isCruella) R.color.cruella_border       else R.color.brand_border)
        if (selected) {
            button.setBackgroundColor(activeBg)
            button.setTextColor(Color.WHITE)
            button.iconTint = ColorStateList.valueOf(Color.WHITE)
            button.strokeWidth = 0
        } else {
            button.setBackgroundColor(inactiveBg)
            button.setTextColor(inactiveText)
            button.iconTint = ColorStateList.valueOf(inactiveText)
            button.strokeWidth = dp(1)
            button.strokeColor = ColorStateList.valueOf(borderCol)
        }
    }

    private fun dp(value: Int): Int =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, value.toFloat(), resources.displayMetrics).toInt()

    private fun dp(value: Float): Float =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, value, resources.displayMetrics)

    data class RecommendationItem(
        val id: String,
        val name: String,
        val category: String,
        val price: String,
        val reason: String,
        val imageUrl: String?,
        val brand: String?,
        val description: String?,
        val sku: String?,
        val stockStatus: String?,
        val sizes: List<String>,
        val metadata: LinkedHashMap<String, String>,
        val itemId: Int? = null,
        val size: String? = null,
    ) {
        fun toJson(): JSONObject = JSONObject().apply {
            put("id", id)
            put("name", name)
            put("category", category)
            put("price", price)
            put("reason", reason)
            imageUrl?.let { put("image_url", it) }
            brand?.let { put("brand", it) }
            description?.let { put("description", it) }
            sku?.let { put("sku", it) }
            stockStatus?.let { put("stock_status", it) }
            put("sizes", JSONArray(sizes))
            put("metadata", JSONObject(metadata as Map<*, *>))
            itemId?.let { put("item_id", it) }
            size?.let { put("size", it) }
        }

        companion object {
            fun fromJson(json: JSONObject): RecommendationItem {
                val metadata = linkedMapOf<String, String>()
                val metaJson = json.optJSONObject("metadata")
                if (metaJson != null) {
                    val keys = metaJson.keys()
                    while (keys.hasNext()) {
                        val key = keys.next()
                        val value = metaJson.optString(key).trim()
                        if (value.isNotBlank()) {
                            metadata[key] = value
                        }
                    }
                }

                val sizes = mutableListOf<String>()
                val sizesJson = json.optJSONArray("sizes")
                if (sizesJson != null) {
                    for (i in 0 until sizesJson.length()) {
                        val value = sizesJson.optString(i).trim()
                        if (value.isNotBlank()) sizes += value
                    }
                }

                return RecommendationItem(
                    id = json.optString("id", ""),
                    name = json.optString("name", "Unnamed item"),
                    category = json.optString("category", "item"),
                    price = json.optString("price", "N/A"),
                    reason = json.optString("reason", ""),
                    imageUrl = json.optString("image_url").takeIf { it.isNotBlank() },
                    brand = json.optString("brand").takeIf { it.isNotBlank() },
                    description = json.optString("description").takeIf { it.isNotBlank() },
                    sku = json.optString("sku").takeIf { it.isNotBlank() },
                    stockStatus = json.optString("stock_status").takeIf { it.isNotBlank() },
                    sizes = sizes,
                    metadata = metadata,
                    itemId = if (json.has("item_id")) json.optInt("item_id") else null,
                    size = json.optString("size").takeIf { it.isNotBlank() },
                )
            }

            /**
             * Maps an agent recommendation item (fields: rank, item_id, size, color,
             * type, brand, price, agent_scores) to a RecommendationItem.
             * Mirrors the web `formatAgentRec` (frontend/js/ui/recommendations.js).
             */
            fun fromAgentJson(json: JSONObject): RecommendationItem {
                val itemId = json.optInt("item_id")
                val typeName = json.optString("type").replace("_", " ").trim()
                val brandName = json.optString("brand").trim()
                val name = listOf(brandName, typeName)
                    .filter { it.isNotBlank() }
                    .joinToString(" ")
                    .ifBlank { "Item $itemId" }

                val price = if (json.has("price") && !json.isNull("price")) {
                    val p = json.optDouble("price", Double.NaN)
                    if (p.isNaN()) "N/A" else "EUR %.2f".format(p)
                } else "N/A"

                val metadata = linkedMapOf<String, String>()
                for (key in listOf(
                    "color", "style", "pattern", "material", "fit",
                    "season", "occasion", "gender", "age_group", "size",
                )) {
                    val value = json.optString(key).trim()
                    if (value.isNotBlank()) metadata[key] = value
                }

                val size = json.optString("size").takeIf { it.isNotBlank() }

                val scores = json.optJSONObject("agent_scores")
                val reason = if (scores != null) {
                    "Agent rank ${json.optInt("rank")} · body ${"%.2f".format(scores.optDouble("body", 0.0))} · " +
                        "clothing ${"%.2f".format(scores.optDouble("clothing", 0.0))} · " +
                        "colour ${"%.2f".format(scores.optDouble("colour", 0.0))} · " +
                        "stock ${"%.2f".format(scores.optDouble("stock", 0.0))}"
                } else ""

                return RecommendationItem(
                    id = itemId.toString(),
                    name = name,
                    category = json.optString("type"),
                    price = price,
                    reason = reason,
                    imageUrl = null,
                    brand = brandName.takeIf { it.isNotBlank() },
                    description = null,
                    sku = null,
                    stockStatus = null,
                    sizes = listOfNotNull(size),
                    metadata = metadata,
                    itemId = itemId,
                    size = size,
                )
            }
        }
    }

    private fun loadPersona(): String =
        getSharedPreferences("viralytics_mobile", Context.MODE_PRIVATE)
            .getString("persona", "").orEmpty()

    private fun savePersona(persona: String) {
        getSharedPreferences("viralytics_mobile", Context.MODE_PRIVATE)
            .edit().putString("persona", persona).apply()
    }

    private fun publishPersona(persona: String) {
        Thread {
            try {
                val payload = JSONObject().put("persona", persona).toString()
                mqttClient?.publish("cruzr/persona", MqttMessage(payload.toByteArray()).apply { qos = 1 })
            } catch (_: Exception) {}
        }.start()
    }

    private fun applyPersona(persona: String) {
        viewModel.selectedPersona = persona
        binding.personaChip?.text = if (persona == "edna") "EDNA active" else "CRUELLA active"
        binding.personaRow?.isVisible = true
        applyPersonaTheme(persona)
    }

    private fun applyPersonaTheme(persona: String) {
        val isCruella = persona == "cruella"

        val bg          = ContextCompat.getColor(this, if (isCruella) R.color.cruella_bg          else R.color.brand_bg)
        val surface     = ContextCompat.getColor(this, if (isCruella) R.color.cruella_surface      else R.color.brand_surface)
        val surfaceSoft = ContextCompat.getColor(this, if (isCruella) R.color.cruella_surface_soft else R.color.brand_surface_soft)
        val textCol     = ContextCompat.getColor(this, if (isCruella) R.color.cruella_text         else R.color.brand_text)
        val muted       = ContextCompat.getColor(this, if (isCruella) R.color.cruella_muted        else R.color.brand_muted)
        val accent      = ContextCompat.getColor(this, if (isCruella) R.color.cruella_accent        else R.color.brand_accent)
        val accentText  = ContextCompat.getColor(this, if (isCruella) R.color.cruella_accent_strong else R.color.brand_accent)
        val border      = ContextCompat.getColor(this, if (isCruella) R.color.cruella_border        else R.color.brand_border)
        val btnPrimary  = if (isCruella) accent else textCol

        // System chrome
        window.statusBarColor = bg
        window.navigationBarColor = bg
        @Suppress("DEPRECATION")
        window.decorView.systemUiVisibility = if (isCruella)
            window.decorView.systemUiVisibility and View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR.inv()
        else
            window.decorView.systemUiVisibility or View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR

        // Root + tab bar backgrounds
        binding.root.setBackgroundColor(bg)
        binding.tabBar?.backgroundTintList = ColorStateList.valueOf(surfaceSoft)

        // Walk the view tree: cards + text (accentText is brighter than accent for legibility on dark bg)
        tintViews(binding.root, textCol, muted, accentText, surface, border)

        // ID-specific overrides for muted elements (walk defaults them to textCol)
        binding.sessionText?.setTextColor(muted)
        binding.statusText?.setTextColor(textCol)
        binding.recommendationsEmptyText?.setTextColor(muted)
        binding.personaChip?.backgroundTintList = ColorStateList.valueOf(accent)

        // Buttons (MaterialButton/ImageButton are skipped in tintViews, handled here)
        binding.captureButton.backgroundTintList = ColorStateList.valueOf(btnPrimary)
        binding.captureButton.setTextColor(Color.WHITE)
        binding.switchPersonaButton?.setTextColor(muted)
        binding.connectionSettingsButton.imageTintList = ColorStateList.valueOf(textCol)
        binding.recommendationsLeftButton.imageTintList = ColorStateList.valueOf(textCol)
        binding.recommendationsRightButton.imageTintList = ColorStateList.valueOf(textCol)
        binding.sendChatButton.backgroundTintList = ColorStateList.valueOf(accent)

        // Chat input
        binding.chatInputLayout?.let { til ->
            til.boxStrokeColor = border
            til.hintTextColor = ColorStateList.valueOf(muted)
            til.defaultHintTextColor = ColorStateList.valueOf(muted)
            til.setBoxBackgroundColorResource(if (isCruella) R.color.cruella_surface_soft else R.color.brand_surface_soft)
        }
        binding.chatInput?.setTextColor(textCol)
        binding.chatInput?.setHintTextColor(muted)

        // Retheme fixed-color drawable backgrounds (shapes baked in brand colors in XML).
        // We replace them with equivalent programmatic shapes so fill + stroke match persona.
        fun roundRect(fill: Int, strokeCol: Int, cornerDp: Float): GradientDrawable =
            GradientDrawable().apply {
                cornerRadius = TypedValue.applyDimension(
                    TypedValue.COMPLEX_UNIT_DIP, cornerDp, resources.displayMetrics)
                setColor(fill)
                setStroke(dp(1), strokeCol)
            }
        binding.statusPill?.background               = roundRect(surface,     border, 999f)
        binding.modeIndicatorText?.background        = roundRect(surfaceSoft, border, 20f)
        binding.chatReplyText?.background            = roundRect(surfaceSoft, border, 20f)
        binding.scanWaitingText?.background          = roundRect(surfaceSoft, border, 22f)
        binding.recommendationsEmptyText?.background = roundRect(surfaceSoft, border, 22f)
        binding.resultImage?.background              = roundRect(surfaceSoft, border, 24f)

        // Re-style tab buttons with new persona colors
        switchTab(currentTab)
    }

    private fun tintViews(view: View, textCol: Int, muted: Int, accent: Int, surface: Int, border: Int) {
        when {
            view.id == R.id.personaChip -> Unit
            view is MaterialCardView -> {
                view.setCardBackgroundColor(surface)
                view.strokeColor = border
            }
            view is MaterialButton || view is android.widget.ImageButton -> Unit
            view is TextView -> when (view.tag?.toString()) {
                "color_muted"  -> view.setTextColor(muted)
                "color_accent" -> view.setTextColor(accent)
                else           -> view.setTextColor(textCol)
            }
        }
        if (view is ViewGroup) {
            for (i in 0 until view.childCount) tintViews(view.getChildAt(i), textCol, muted, accent, surface, border)
        }
    }

    private fun showPersonaDialog() {
        val options = arrayOf(
            "Cruella — YOLO vision + LLM text + strict matching",
            "Edna — FashionNet vision + custom text + flexible matching",
        )
        MaterialAlertDialogBuilder(this)
            .setTitle(getString(R.string.persona_dialog_title))
            .setCancelable(loadPersona().isNotBlank())
            .setItems(options) { _, which ->
                val persona = if (which == 0) "cruella" else "edna"
                savePersona(persona)
                applyPersona(persona)
                publishPersona(persona)
                viewModel.clearSession()
                renderDetections()
                renderRecommendations()
            }
            .show()
    }

    private fun startMqttListener() {
        Thread {
            val brokerUri = "ssl://test.mosquitto.org:8883"
            val clientId = "Cruzr_${(1000..9999).random()}"

            try {
                mqttClient = MqttClient(brokerUri, clientId, MemoryPersistence())

                val trustAll = arrayOf<javax.net.ssl.TrustManager>(object : javax.net.ssl.X509TrustManager {
                    override fun checkClientTrusted(chain: Array<java.security.cert.X509Certificate>, authType: String) {}
                    override fun checkServerTrusted(chain: Array<java.security.cert.X509Certificate>, authType: String) {}
                    override fun getAcceptedIssuers(): Array<java.security.cert.X509Certificate> = arrayOf()
                })
                val sslCtx = javax.net.ssl.SSLContext.getInstance("TLS")
                sslCtx.init(null, trustAll, java.security.SecureRandom())

                val options = MqttConnectOptions().apply {
                    isCleanSession = true
                    connectionTimeout = 10
                    isAutomaticReconnect = true
                    socketFactory = sslCtx.socketFactory
                }

                mqttClient?.setCallback(object : MqttCallbackExtended {
                    override fun connectComplete(reconnect: Boolean, serverURI: String?) {
                        if (!reconnect) return
                        try {
                            mqttClient?.subscribe("cruzr/persona", 1)
                            mqttClient?.subscribe("cruzr/commands", 1)
                            mqttClient?.subscribe("cruzr/scan_result", 1)
                        } catch (_: Exception) {}
                        mqttReconnectHandler.removeCallbacks(mqttReconnectRunnable)
                        runOnUiThread { setStatus("MQTT Reconnected to test.mosquitto.org") }
                    }

                    override fun connectionLost(cause: Throwable?) {
                        runOnUiThread { setStatus("MQTT lost — reconnecting…") }
                        mqttReconnectHandler.postDelayed(mqttReconnectRunnable, 15_000L)
                    }

                    override fun messageArrived(topic: String?, message: MqttMessage?) {
                        val payloadString = message?.toString() ?: return
                        try {
                            val json = JSONObject(payloadString)

                            if (topic == "cruzr/persona") {
                                val p = json.optString("persona").ifBlank { null } ?: return
                                runOnUiThread { applyPersona(p); savePersona(p) }
                                return
                            }

                            if (topic == "cruzr/scan_result") {
                                handleScanResult(json)
                                return
                            }

                            val action = json.optString("action")

                            when (action) {
                                "move_to_stand" -> {
                                    val target = json.optString("target", "unknown_rack")
                                    runOnUiThread { toast("PC: navigate to $target") }
                                    sendRobotNavigationCommand(target, "AI recommendation received via network")
                                }
                                "move_to_coords" -> {
                                    val x = json.optDouble("x", 0.0).toFloat()
                                    val y = json.optDouble("y", 0.0).toFloat()
                                    val theta = json.optDouble("theta", 0.0).toFloat()
                                    runOnUiThread { toast("PC: navigate to coords $x, $y") }
                                    sendRawCoordinateCommand(x, y, theta)
                                }
                                "speak" -> {
                                    val text = json.optString("text", "")
                                    runOnUiThread { setStatus("Speaking: $text") }
                                    speakText(text)
                                }
                                "guide_user" -> {
                                    val target = json.optString("target", "")
                                    val introText = json.optString("intro_text", "")
                                    runOnUiThread { setStatus("Guide: going to '$target'") }
                                    speakText(introText)
                                    sendRobotNavigationCommand(target, "", skipIntroSpeech = true)
                                }
                                "stop_navigation" -> {
                                    runOnUiThread { setStatus("Stop requested (unsupported by SDK)") }
                                    publishStatus("error", JSONObject().put("message", "stop_navigation not supported by SDK v2.8.0"))
                                }
                                "locate_self" -> {
                                    val nav = navigationManager
                                    if (nav == null) {
                                        publishStatus("error", JSONObject().put("message", "Navigation Manager offline"))
                                    } else if (nav.isSelfLocated) {
                                        val payload = JSONObject().put("self_located", true).put("note", "already located")
                                        try {
                                            val loc = nav.currentLocation
                                            payload.put("x", loc.position.x)
                                            payload.put("y", loc.position.y)
                                            payload.put("theta", loc.rotation)
                                        } catch (e: Exception) {}
                                        publishStatus("status_report", payload)
                                    } else {
                                        runOnUiThread { setStatus("Locating self…") }
                                        publishStatus("localization_started", null)
                                        nav.locateSelf()
                                            .done {
                                                runOnUiThread { setStatus("Self-located.") }
                                                val payload = JSONObject()
                                                try {
                                                    val loc = nav.currentLocation
                                                    payload.put("x", loc.position.x)
                                                    payload.put("y", loc.position.y)
                                                    payload.put("theta", loc.rotation)
                                                } catch (e: Exception) {}
                                                publishStatus("localization_success", payload)
                                            }
                                            .fail { error ->
                                                val errMsg = error?.message ?: "unknown"
                                                runOnUiThread { setStatus("Localization failed: $errMsg") }
                                                publishStatus("localization_failed", JSONObject().put("error_message", errMsg))
                                            }
                                    }
                                }
                                "get_status" -> {
                                    val nav = navigationManager
                                    val payload = JSONObject().apply {
                                        put("nav_manager_online", nav != null)
                                        if (nav != null) {
                                            put("self_located", nav.isSelfLocated)
                                            try {
                                                val loc = nav.currentLocation
                                                put("x", loc.position.x)
                                                put("y", loc.position.y)
                                                put("theta", loc.rotation)
                                            } catch (e: Exception) {}
                                        }
                                    }
                                    publishStatus("status_report", payload)
                                }
                                "greet" -> {
                                    if (isBusy || isServingCustomer) {
                                        publishStatus("robot_busy", JSONObject()
                                            .put("command", "greet")
                                            .put("reason", if (isServingCustomer) "robot is serving a customer" else "robot is currently busy"))
                                        runOnUiThread { setStatus("Greet ignored — robot is busy") }
                                    } else {
                                        val x       = json.optDouble("x", 0.0).toFloat()
                                        val y       = json.optDouble("y", 0.0).toFloat()
                                        val theta   = json.optDouble("theta", 0.0).toFloat()
                                        entranceX = x
                                        entranceY = y
                                        entranceTheta = theta
                                        entranceCoordsSet = true
                                        pendingGreetText    = json.optString("text", pendingGreetText)
                                        pendingGreetGesture = json.optString("gesture", "wave")
                                        runOnUiThread { setStatus("Greeting sequence: moving to entrance…") }
                                        sendGreetCommand(x, y, theta)
                                    }
                                }
                                "gesture" -> {
                                    val name = json.optString("name", "wave")
                                    doGesture(name)
                                }
                                "farewell" -> {
                                    if (!isBusy && !isServingCustomer) {
                                        val text = json.optString("text", "Goodbye! Hope to see you again soon!")
                                        val gesture = json.optString("gesture", "goodbye")
                                        speakText(text)
                                        Handler(Looper.getMainLooper()).postDelayed({
                                            doGesture(gesture)
                                        }, 1000L)
                                    }
                                }
                            }
                        } catch (e: Exception) {
                            e.printStackTrace()
                        }
                    }

                    override fun deliveryComplete(token: IMqttDeliveryToken?) {}
                })

                mqttClient?.connect(options)
                mqttClient?.subscribe("cruzr/persona", 1)
                mqttClient?.subscribe("cruzr/commands", 1)
                mqttClient?.subscribe("cruzr/scan_result", 1)
                mqttReconnectHandler.removeCallbacks(mqttReconnectRunnable)
                runOnUiThread { setStatus("MQTT Connected to test.mosquitto.org") }

            } catch (e: Exception) {
                runOnUiThread { setStatus("MQTT Setup Failed: ${e.message}") }
            }
        }.start()
    }

    private fun handleScanResult(payload: JSONObject) {
        val sessionId = payload.optString("session_id").ifBlank { null }
        val persona = payload.optString("persona").ifBlank { null }

        val detections = mutableListOf<String>()
        val detArray = payload.optJSONArray("detections")
        if (detArray != null) {
            for (i in 0 until detArray.length()) {
                val name = detArray.optJSONObject(i)?.optString("class_name")?.trim() ?: continue
                if (name.isNotBlank()) detections.add(name)
            }
        }

        val recommendations = mutableListOf<RecommendationItem>()
        val recArray = payload.optJSONArray("recommendations")
        if (recArray != null) {
            for (i in 0 until recArray.length()) {
                val item = recArray.optJSONObject(i) ?: continue
                recommendations.add(RecommendationItem.fromJson(item))
            }
        }

        val annotatedFrame = payload.optString("annotated_frame").ifBlank { null }

        val detectedColor = detArray?.optJSONObject(0)?.optString("color_name")?.trim().orEmpty()
        val detectedBodyType = payload.optJSONObject("body_analysis")
            ?.optString("body_shape")?.trim()
            ?.takeIf { it.isNotBlank() && it != "unknown" }
            .orEmpty()
        val baseUrl = normalizedBaseUrl()

        viewModel.injectScanResult(
            sessionId, detections, recommendations, annotatedFrame,
            detectedColor, detectedBodyType, baseUrl,
        )
        if (persona != null) {
            runOnUiThread { applyPersona(persona) }
        }
    }

    private fun initCruzrHardware() {
        setStatus("Initializing V2.8.0 Hardware Managers...")
        try {
            navigationManager  = Robot.globalContext().getSystemService(NavigationManager.SERVICE) as NavigationManager
            speechManager      = Robot.globalContext().getSystemService("speech") as SpeechManager
            motionManager      = Robot.globalContext().getSystemService("motion") as? MotionManager
            cruzrSensorManager = Robot.globalContext().getSystemService(CruzrSensorManager.SERVICE) as? CruzrSensorManager

            val motionOk = if (motionManager  != null) "✓" else "✗"
            val sensorOk = if (cruzrSensorManager != null) "✓" else "✗"
            runOnUiThread { setStatus("Hardware ready — motion:$motionOk sensor:$sensorOk") }
        } catch (e: Exception) {
            runOnUiThread { setStatus("Manager Init Failed: ${e.message}") }
        }
    }

    private fun publishStatus(event: String, extras: JSONObject?) {
        val payload = JSONObject().apply {
            put("event", event)
            put("timestamp", System.currentTimeMillis() / 1000L)
            extras?.keys()?.forEach { key -> put(key, extras.get(key)) }
        }
        try {
            val msg = MqttMessage(payload.toString().toByteArray(Charsets.UTF_8)).apply { qos = 1 }
            mqttClient?.publish("cruzr/status", msg)
        } catch (e: Exception) {
            android.util.Log.w("CruzrApp", "publishStatus failed: ${e.message}")
        }
    }

    private fun speakText(textToSay: String) {
        if (speechManager != null) {
            try {
                speechManager?.synthesize(textToSay)?.fail { error ->
                    android.util.Log.e("CruzrApp", "Native voice failed: ${error.message}. Using fallback.")
                    textToSpeech.speak(textToSay, TextToSpeech.QUEUE_FLUSH, null, null)
                }
            } catch (e: Exception) {
                android.util.Log.e("CruzrApp", "Native voice crashed. Using fallback.")
                textToSpeech.speak(textToSay, TextToSpeech.QUEUE_FLUSH, null, null)
            }
        } else {
            textToSpeech.speak(textToSay, TextToSpeech.QUEUE_FLUSH, null, null)
        }
    }

    private fun sendRawCoordinateCommand(x: Float, y: Float, theta: Float) {
        val nav = navigationManager
        if (nav == null) {
            publishStatus("error", JSONObject().put("message", "Navigation Manager offline"))
            return
        }

        isBusy = true
        publishStatus("navigation_started", JSONObject().put("target_type", "raw_coords").put("x", x).put("y", y))
        runOnUiThread { setStatus("Moving to raw coordinates $x, $y...") }

        try {
            val destination = Location.Builder(Point(x, y)).setRotation(theta).build()
            val option = NavigationOption.Builder(destination)
                .setTrackMode(false)
                .setMaxSpeed(NAV_MAX_SPEED)
                .setRetryCount(NAV_RETRY_COUNT)
                .setRetryInterval(NAV_RETRY_INTERVAL)
                .build()

            startActivity(Intent(this@MainActivity, MainActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_REORDER_TO_FRONT)
            })

            Handler(Looper.getMainLooper()).postDelayed(Runnable {
                nav.navigate(option)
                    .done {
                        isBusy = false
                        publishStatus("navigation_arrived", JSONObject().put("target_type", "raw_coords"))
                        runOnUiThread { setStatus("Arrived at coordinates!") }
                    }
                    .fail { error ->
                        val errCode = error?.code ?: -1
                        val errMsg = error?.message ?: "unknown"
                        isBusy = false
                        publishStatus("navigation_failed", JSONObject()
                            .put("error_code", errCode)
                            .put("error_message", errMsg))
                        runOnUiThread { setStatus("Coord Navigation jammed: $errMsg") }
                    }
            }, 500L)

        } catch (e: Exception) {
            android.util.Log.e("CruzrApp", "CRASH routing raw coords: ${e.message}")
        }
    }

    private fun sendRobotNavigationCommand(targetItem: String, reason: String, skipIntroSpeech: Boolean = false) {
        val nav = navigationManager
        if (nav == null) {
            runOnUiThread { setStatus("Navigation Manager offline.") }
            publishStatus("error", JSONObject().put("message", "Navigation Manager offline"))
            return
        }

        isBusy = true
        publishStatus("navigation_started", JSONObject().put("target", targetItem))

        if (!skipIntroSpeech) {
            val textToSpeak = if (reason.isNotBlank()) {
                "Encontrei uma ótima opção! $reason. Siga-me, vou mostrar-lhe onde está."
            } else {
                "Siga-me, vou mostrar-lhe onde fica o $targetItem."
            }
            speakText(textToSpeak)
        }

        runOnUiThread { setStatus("Checking spatial location...") }

        if (!nav.isSelfLocated) {
            runOnUiThread { setStatus("Locating self in the room...") }
            publishStatus("localization_started", null)

            nav.locateSelf().done {
                runOnUiThread { Toast.makeText(this@MainActivity, "Localized!", Toast.LENGTH_SHORT).show() }
                publishStatus("localization_success", null)
                findMarkerAndNavigate(nav, targetItem)
            }.fail { error ->
                val errMsg = error?.message ?: "unknown"
                isBusy = false
                runOnUiThread { setStatus("Failed to localize: $errMsg") }
                publishStatus("localization_failed", JSONObject().put("error_message", errMsg))
            }
        } else {
            findMarkerAndNavigate(nav, targetItem)
        }
    }

    private fun findMarkerAndNavigate(nav: NavigationManager, targetItem: String) {
        // Resolve clothing category codes → stand name, then look up surveyed coordinates.
        val resolvedTarget = CATEGORY_STAND_MAP[targetItem.lowercase()] ?: targetItem
        android.util.Log.d("CruzrNav", "Target '$targetItem' → stand '$resolvedTarget'")

        val entry = standCoordinates.entries.find { it.key.equals(resolvedTarget, ignoreCase = true) }
        if (entry == null) {
            isBusy = false
            runOnUiThread { setStatus("No coordinates for '$resolvedTarget'. Add it to coordinates.json.") }
            publishStatus("nav_no_coords", JSONObject().put("target", resolvedTarget).put("raw_category", targetItem))
            return
        }

        val (x, y, theta) = entry.value
        android.util.Log.d("CruzrNav", "Navigating to '${entry.key}' @ x=$x y=$y theta=$theta (trackMode=$TRACK_MODE)")
        runOnUiThread { setStatus("Navigating to ${entry.key}...") }

        try {
            val location = Location.Builder(Point(x, y))
                .setRotation(theta)
                .build()
            val option = NavigationOption.Builder(location)
                .setTrackMode(TRACK_MODE)
                .setMaxSpeed(NAV_MAX_SPEED)
                .setRetryCount(NAV_RETRY_COUNT)
                .setRetryInterval(NAV_RETRY_INTERVAL)
                .build()

            startActivity(Intent(this@MainActivity, MainActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_REORDER_TO_FRONT)
            })

            Handler(Looper.getMainLooper()).postDelayed({
                nav.navigate(option)
                    .done {
                        android.util.Log.d("CruzrNav", "Navigation DONE — arrived at ${entry.key}!")
                        isBusy = false
                        publishStatus("navigation_arrived", JSONObject().put("target", entry.key))
                        runOnUiThread {
                            setStatus("Arrived at ${entry.key}!")
                            speakText("Aqui está o artigo que procurava!")
                        }
                        endCustomerSession("arrived_at_stand")
                    }
                    .progress { p ->
                        android.util.Log.d("CruzrNav", "Nav progress: $p")
                        publishStatus("navigation_progress", JSONObject().put("target", entry.key).put("data", p.toString()))
                    }
                    .fail { error ->
                        val errCode = error?.code ?: -1
                        val errMsg = error?.message ?: "unknown"
                        android.util.Log.e("CruzrNav", "Navigation FAILED: $errMsg / code: $errCode")
                        isBusy = false
                        publishStatus("navigation_failed", JSONObject()
                            .put("target", entry.key)
                            .put("error_code", errCode)
                            .put("error_message", errMsg))
                        runOnUiThread { setStatus("Navigation failed: $errMsg (code $errCode)") }
                    }
            }, 500L)

        } catch (e: Exception) {
            isBusy = false
            android.util.Log.e("CruzrNav", "Nav crash: ${e.message}")
            runOnUiThread { setStatus("Nav crash: ${e.message}") }
        }
    }

    private fun loadStandCoordinates(): Map<String, Triple<Float, Float, Float>> {
        return try {
            val json = JSONObject(assets.open("coordinates.json").bufferedReader().readText())
            buildMap {
                json.keys().forEach { key ->
                    val obj = json.optJSONObject(key) ?: return@forEach
                    val x     = obj.optDouble("x",     0.0).toFloat()
                    val y     = obj.optDouble("y",     0.0).toFloat()
                    val theta = obj.optDouble("theta", 0.0).toFloat()
                    put(key, Triple(x, y, theta))
                }
            }
        } catch (e: Exception) {
            android.util.Log.e("CruzrNav", "Failed to load coordinates.json: ${e.message}")
            emptyMap()
        }
    }

    private fun sendGreetCommand(x: Float, y: Float, theta: Float) {
        val nav = navigationManager ?: run {
            publishStatus("error", JSONObject().put("message", "Navigation Manager offline"))
            isBusy = false
            return
        }

        isBusy = true
        isAtEntrance = false
        publishStatus("greeting_started", JSONObject().put("x", x).put("y", y))

        try {
            val destination = Location.Builder(Point(x, y)).setRotation(theta).build()
            val option = NavigationOption.Builder(destination)
                .setTrackMode(TRACK_MODE)
                .setMaxSpeed(NAV_MAX_SPEED)
                .setRetryCount(NAV_RETRY_COUNT)
                .setRetryInterval(NAV_RETRY_INTERVAL)
                .build()

            startActivity(Intent(this@MainActivity, MainActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_REORDER_TO_FRONT)
            })

            Handler(Looper.getMainLooper()).postDelayed({
                nav.navigate(option)
                    .done {
                        runOnUiThread { setStatus("At entrance — LIDAR watching for customer…") }
                        publishStatus("greeting_at_entrance", null)
                        isAtEntrance = true
                        startHumanDetection()
                    }
                    .fail { error ->
                        val errMsg = error?.message ?: "unknown"
                        publishStatus("navigation_failed", JSONObject()
                            .put("context", "greet")
                            .put("error_message", errMsg))
                        runOnUiThread { setStatus("Greeting nav failed: $errMsg") }
                        isBusy = false
                    }
            }, 500L)

        } catch (e: Exception) {
            android.util.Log.e("CruzrApp", "sendGreetCommand crash: ${e.message}")
            isBusy = false
        }
    }

    // ── LIDAR human detection ─────────────────────────────────────────────────

    private fun startHumanDetection() {
        val sm = cruzrSensorManager ?: run {
            android.util.Log.w("CruzrApp", "SensorManager not available — LIDAR detection disabled")
            onPersonDetectedByLidar()
            return
        }
        try {
            humanDetectListener = object : SensorListener {
                override fun onSensorChanged(device: SensorDevice, event: SensorEvent) {
                    if (isAtEntrance) onPersonDetectedByLidar()
                }
            }
            sm.registerListener("human_detect", humanDetectListener!!)
            android.util.Log.i("CruzrApp", "LIDAR human detection active")
            lidarWaitHandler.postDelayed(lidarWaitTimeoutRunnable, LIDAR_WAIT_TIMEOUT_MS)
        } catch (e: Exception) {
            android.util.Log.e("CruzrApp", "registerListener failed: ${e.message}")
            onPersonDetectedByLidar()
        }
    }

    private fun stopHumanDetection() {
        val sm = cruzrSensorManager ?: return
        val listener = humanDetectListener ?: return
        try {
            sm.unregisterListener(listener)
        } catch (e: Exception) {
            android.util.Log.w("CruzrApp", "unregisterListener failed: ${e.message}")
        }
        humanDetectListener = null
    }

    private fun onPersonDetectedByLidar() {
        if (!isAtEntrance) return
        isAtEntrance = false
        lidarWaitHandler.removeCallbacks(lidarWaitTimeoutRunnable)
        stopHumanDetection()

        runOnUiThread { setStatus("Customer detected! Greeting…") }
        publishStatus("person_detected", JSONObject().put("source", "lidar"))

        speakText(pendingGreetText)

        Handler(Looper.getMainLooper()).postDelayed({
            doGesture(pendingGreetGesture)
        }, 1500L)

        isServingCustomer = true
        sessionHandler.postDelayed(sessionTimeoutRunnable, SESSION_TIMEOUT_MS)


        Handler(Looper.getMainLooper()).postDelayed({
            isBusy = false
            publishStatus("greeting_complete", JSONObject().put("ready_for_scan", true))
            runOnUiThread { setStatus("Greeting done — ready for customer scan") }
        }, 6000L)
    }

    // ── Customer session ──────────────────────────────────────────────────────

    private fun extendSession() {
        if (!isServingCustomer) return
        sessionHandler.removeCallbacks(sessionTimeoutRunnable)
        sessionHandler.postDelayed(sessionTimeoutRunnable, SESSION_TIMEOUT_MS)
    }

    private fun endCustomerSession(reason: String) {
        if (!isServingCustomer) return
        sessionHandler.removeCallbacks(sessionTimeoutRunnable)
        isServingCustomer = false
        publishStatus("session_ended", JSONObject().put("reason", reason))
        runOnUiThread { setStatus("Session ended ($reason) — returning to entrance") }
        returnToEntrance()
    }

    private fun returnToEntrance() {
        if (!entranceCoordsSet || isBusy) return
        sendRawCoordinateCommand(entranceX, entranceY, entranceTheta)
    }

    // ── Gestures ──────────────────────────────────────────────────────────────

    private fun doGesture(gestureName: String) {
        val mm = motionManager ?: run {
            android.util.Log.w("CruzrApp", "MotionManager offline — gesture '$gestureName' skipped")
            publishStatus("gesture_failed", JSONObject()
                .put("gesture", gestureName).put("reason", "MotionManager offline"))
            return
        }

        val actionId = when (gestureName.lowercase().replace("-", "_")) {
            "wave"        -> "swingarm"
            "raise"       -> "zhanggao"
            "handshake"   -> "shankhand"
            "guide_left"  -> "guideleft"
            "guide_right" -> "guideright"
            "applause"    -> "applause"
            "surprise"    -> "surprise"
            "goodbye"     -> "goodbye"
            "searching"   -> "searching"
            "cute"        -> "cute"
            "reset"       -> "RESET"
            else          -> gestureName
        }

        try {
            val uri    = Uri.parse("action://ubtrobot/$actionId")
            val option = PerformingOption.Builder(uri).build()
            mm.performAction(option)
                .done {
                    publishStatus("gesture_performed", JSONObject().put("gesture", gestureName))
                }
                .fail { error ->
                    android.util.Log.e("CruzrApp", "Gesture '$gestureName' failed: ${error?.message}")
                    publishStatus("gesture_failed", JSONObject()
                        .put("gesture", gestureName)
                        .put("error", error?.message ?: "unknown"))
                }
        } catch (e: Exception) {
            android.util.Log.e("CruzrApp", "doGesture crash for '$gestureName': ${e.message}")
        }
    }

    private fun Bitmap.applyExifRotation(path: String): Bitmap {
        val degrees = when (ExifInterface(path).getAttributeInt(
            ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
        )) {
            ExifInterface.ORIENTATION_ROTATE_90  -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> return this
        }
        val m = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, m, true).also { recycle() }
    }

    override fun onDestroy() {
        isAtEntrance = false
        isServingCustomer = false
        sessionHandler.removeCallbacks(sessionTimeoutRunnable)
        mqttReconnectHandler.removeCallbacks(mqttReconnectRunnable)
        stopHumanDetection()
        try { mqttClient?.disconnect() } catch (_: Exception) {}
        if (::textToSpeech.isInitialized) {
            textToSpeech.stop()
            textToSpeech.shutdown()
        }
        super.onDestroy()
    }
}
