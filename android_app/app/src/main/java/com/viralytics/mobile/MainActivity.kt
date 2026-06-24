package com.viralytics.mobile

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
import org.eclipse.paho.client.mqttv3.MqttCallback
import org.eclipse.paho.client.mqttv3.IMqttDeliveryToken
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val viewModel: MainViewModel by viewModels()

    // --- NAVIGATION CONFIG ---
    private val TRACK_MODE = false
    private val NAV_MAX_SPEED = 0.5f
    private val NAV_RETRY_COUNT = 2
    private val NAV_RETRY_INTERVAL = 2000

    private var currentTab: String = "scan"

    private var mqttClient: MqttClient? = null

    // Hardware Managers (V2.8.0 Architecture)
    private var navigationManager: NavigationManager? = null
    private var speechManager: SpeechManager? = null
    private var motionManager: MotionManager? = null
    private var cruzrSensorManager: CruzrSensorManager? = null
    private lateinit var textToSpeech: TextToSpeech

    // Greeting flow state — guards against duplicate door triggers
    @Volatile private var isBusy = false
    @Volatile private var isAtEntrance = false
    private var humanDetectListener: SensorListener? = null
    @Volatile private var pendingGreetText  = "Welcome! I'm Cruzr, your personal fashion assistant. Come closer and let me help you find the perfect outfit!"
    @Volatile private var pendingGreetGesture = "wave"

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                cameraLauncher.launch(null)
            } else {
                setStatus("Camera permission denied.")
            }
        }

    private val cameraLauncher =
        registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
            if (bitmap == null) {
                setStatus("Capture cancelled.")
                return@registerForActivityResult
            }
            val baseUrl = normalizedBaseUrl() ?: return@registerForActivityResult
            viewModel.uploadScan(bitmap, baseUrl, loadPersona())
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initCruzrHardware()
        startMqttListener()
        observeViewModel()

        setStatus("Ready.")
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
                    renderDetections()
                    renderRecommendations()
                    updateAnnotatedImage(event.annotatedFrameBase64)
                    switchTab("scan")
                    updateSessionLabel("Vision-led")
                    showChatReply("Scan complete. Tap a recommendation to inspect it, or refine with chat.")
                }
                is UiEvent.ScanError -> showChatReply(event.message)
                is UiEvent.ChatComplete -> {
                    renderRecommendations()
                    switchTab("refine")
                    val mode = if (binding.replaceVisionSwitch.isChecked) "Search-led override" else "Vision + search"
                    updateSessionLabel(mode)
                    showChatReply(event.reply)
                    binding.chatInput.text?.clear()
                }
                is UiEvent.AgentRecsComplete -> {
                    renderRecommendations()
                }
                is UiEvent.ChatError -> showChatReply(event.message)
            }
        }
    }

    private fun launchCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            return
        }
        val btn = binding.captureButton
        btn.isEnabled = false
        object : android.os.CountDownTimer(5000, 1000) {
            override fun onTick(millisUntilFinished: Long) {
                btn.text = "${(millisUntilFinished / 1000) + 1}"
            }
            override fun onFinish() {
                btn.text = getString(R.string.capture_outfit)
                btn.isEnabled = true
                cameraLauncher.launch(null)
            }
        }.start()
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
        return Chip(this).apply {
            text = label.replaceFirstChar { it.uppercase() }
            isClickable = false
            isCheckable = false
            chipBackgroundColor = ContextCompat.getColorStateList(context, R.color.brand_surface_soft)
            chipStrokeColor = ContextCompat.getColorStateList(context, R.color.brand_border)
            chipStrokeWidth = dp(1f)
            setTextColor(ContextCompat.getColor(context, R.color.brand_text))
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

        val card = MaterialCardView(this).apply {
            layoutParams = LinearLayout.LayoutParams(dp(200), LinearLayout.LayoutParams.MATCH_PARENT).also {
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
        val imageView = ImageView(this).apply {
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(120))
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

        MaterialAlertDialogBuilder(this)
            .setView(dialogView)
            .setPositiveButton("Take me there!") { _, _ ->
                sendRobotNavigationCommand(item.category, item.reason)
            }
            .setNegativeButton("Close", null)
            .show()
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

        MaterialAlertDialogBuilder(this)
            .setTitle(getString(R.string.connection_title))
            .setView(dialogView)
            .setPositiveButton(getString(R.string.connection_save)) { _, _ ->
                saveServerUrl(input.text?.toString().orEmpty())
                toast("Connection saved.")
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
        val metadata: LinkedHashMap<String, String>
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
        val accent      = ContextCompat.getColor(this, if (isCruella) R.color.cruella_accent       else R.color.brand_accent)
        val border      = ContextCompat.getColor(this, if (isCruella) R.color.cruella_border       else R.color.brand_border)
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

        // Walk the view tree: cards + text
        tintViews(binding.root, textCol, muted, accent, surface, border)

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
                viewModel.clearSession()
                renderDetections()
                renderRecommendations()
            }
            .show()
    }

    private fun startMqttListener() {
        Thread {
            val rawUrl = loadServerUrl().replace("http://", "").replace("https://", "")
            val ipAddress = rawUrl.split(":")[0]
            val brokerUri = "tcp://$ipAddress:1883"
            val clientId = "Cruzr_${(1000..9999).random()}"

            try {
                mqttClient = MqttClient(brokerUri, clientId, MemoryPersistence())
                val options = MqttConnectOptions().apply {
                    isCleanSession = true
                    connectionTimeout = 10
                    isAutomaticReconnect = true
                }

                mqttClient?.setCallback(object : MqttCallback {
                    override fun connectionLost(cause: Throwable?) {
                        runOnUiThread { setStatus("MQTT lost — reconnecting…") }
                    }

                    override fun messageArrived(topic: String?, message: MqttMessage?) {
                        val payloadString = message?.toString() ?: return
                        try {
                            val json = JSONObject(payloadString)
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
                                    if (isBusy) {
                                        publishStatus("robot_busy", JSONObject()
                                            .put("command", "greet")
                                            .put("reason", "robot is currently engaged with a customer"))
                                        runOnUiThread { setStatus("Greet ignored — robot is busy") }
                                    } else {
                                        val x     = json.optDouble("x", 0.0).toFloat()
                                        val y     = json.optDouble("y", 0.0).toFloat()
                                        val theta = json.optDouble("theta", 0.0).toFloat()
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
                            }
                        } catch (e: Exception) {
                            e.printStackTrace()
                        }
                    }

                    override fun deliveryComplete(token: IMqttDeliveryToken?) {}
                })

                mqttClient?.connect(options)
                mqttClient?.subscribe("cruzr/commands")
                runOnUiThread { setStatus("MQTT Connected to $ipAddress") }

            } catch (e: Exception) {
                runOnUiThread { setStatus("MQTT Setup Failed: ${e.message}") }
            }
        }.start()
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
                "I found a great match! $reason. Let me show you where it is."
            } else {
                "Let me show you where the $targetItem is."
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
        runOnUiThread { setStatus("Loading internal map...") }

        nav.currentNavMap.done { navMap ->
            val scale = navMap.scale
            android.util.Log.d("CruzrNav", "=== MAP SCALE = $scale ===")

            try {
                val here = nav.currentLocation
                android.util.Log.d("CruzrNav", "CURRENT LOCATION (meters): x=${here?.position?.x} y=${here?.position?.y} rot=${here?.rotation}")
            } catch (e: Exception) {
                android.util.Log.w("CruzrNav", "getCurrentLocation threw: ${e.message}")
            }

            val polys = navMap.polylineList
            android.util.Log.d("CruzrNav", "=== MAP POLYLINES (${polys?.size ?: 0}) ===")
            polys?.forEachIndexed { i, poly ->
                android.util.Log.d("CruzrNav", " Poly[$i] name='${poly.name}' points=${poly.locationList?.size ?: 0}")
                poly.locationList?.forEachIndexed { j, loc ->
                    android.util.Log.d("CruzrNav", " [$j] x=${loc.position?.x} y=${loc.position?.y}")
                }
            }
            android.util.Log.d("CruzrNav", "=== MAP MARKERS (${navMap.markerList?.size ?: 0}) ===")
            navMap.markerList?.forEach { m ->
                android.util.Log.d("CruzrNav", " Marker '${m.title}' x=${m.position?.x} y=${m.position?.y}")
            }

            val markers = navMap.markerList

            if (markers.isNullOrEmpty()) {
                isBusy = false
                runOnUiThread { setStatus("Map has no markers.") }
                publishStatus("marker_not_found", JSONObject().put("target", targetItem).put("reason", "map has no markers"))
                return@done
            }

            val targetMarker = markers.find { it.title.equals(targetItem, ignoreCase = true) }

            if (targetMarker == null) {
                isBusy = false
                runOnUiThread { setStatus("Marker '$targetItem' not found on map.") }
                publishStatus("marker_not_found", JSONObject().put("target", targetItem))
                return@done
            }

            android.util.Log.d("CruzrNav", "Marker found: ${targetMarker.title} @ ${targetMarker.position}")

            try {
                val option = NavigationOption.Builder(targetMarker)
                    .setTrackMode(TRACK_MODE)
                    .setMaxSpeed(NAV_MAX_SPEED)
                    .setRetryCount(NAV_RETRY_COUNT)
                    .setRetryInterval(NAV_RETRY_INTERVAL)
                    .build()

                android.util.Log.d("CruzrNav", "Navigating (trackMode=$TRACK_MODE) to ${targetMarker.title}")
                runOnUiThread { setStatus("Navigating to $targetItem...") }

                Handler(Looper.getMainLooper()).postDelayed(Runnable {
                    nav.navigate(option)
                        .done {
                            android.util.Log.d("CruzrNav", "Navigation DONE - arrived!")
                            isBusy = false
                            publishStatus("navigation_arrived", JSONObject().put("target", targetItem))
                            runOnUiThread {
                                setStatus("Arrived at $targetItem!")
                                speakText("Here is the item you are looking for.")
                            }
                        }
                        .progress { p ->
                            android.util.Log.d("CruzrNav", "Nav progress: $p")
                            publishStatus("navigation_progress", JSONObject().put("target", targetItem).put("data", p.toString()))
                        }
                        .fail { error ->
                            val errCode = error?.code ?: -1
                            val errMsg = error?.message ?: "unknown"
                            android.util.Log.e("CruzrNav", "Navigation FAILED: $errMsg / code: $errCode")
                            isBusy = false
                            publishStatus("navigation_failed", JSONObject()
                                .put("target", targetItem)
                                .put("error_code", errCode)
                                .put("error_message", errMsg))
                            runOnUiThread { setStatus("Navigation jammed: $errMsg (code $errCode)") }
                        }
                }, 500L)

            } catch (e: Exception) {
                isBusy = false
                android.util.Log.e("CruzrNav", "CRASH building option or navigating: ${e.message}")
                runOnUiThread { setStatus("Nav crash: ${e.message}") }
            }

        }.fail { error ->
            val errMsg = error?.message ?: "unknown"
            isBusy = false
            runOnUiThread { setStatus("Map access failed: $errMsg") }
            publishStatus("error", JSONObject().put("message", "Map access failed: $errMsg"))
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
        stopHumanDetection()

        runOnUiThread { setStatus("Customer detected! Greeting…") }
        publishStatus("person_detected", JSONObject().put("source", "lidar"))

        speakText(pendingGreetText)

        Handler(Looper.getMainLooper()).postDelayed({
            doGesture(pendingGreetGesture)
        }, 1500L)

        Handler(Looper.getMainLooper()).postDelayed({
            isBusy = false
            publishStatus("greeting_complete", JSONObject().put("ready_for_scan", true))
            runOnUiThread { setStatus("Greeting done — ready for customer scan") }
        }, 6000L)
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

    override fun onDestroy() {
        isAtEntrance = false
        stopHumanDetection()
        try { mqttClient?.disconnect() } catch (_: Exception) {}
        if (::textToSpeech.isInitialized) {
            textToSpeech.stop()
            textToSpeech.shutdown()
        }
        super.onDestroy()
    }
}
