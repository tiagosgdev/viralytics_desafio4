package com.viralytics.mobile

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Typeface
import android.os.Bundle
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
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.chip.Chip
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.viralytics.mobile.databinding.ActivityMainBinding
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream

// V2.8.0 UBTech Robot Imports
import com.ubtrobot.Robot
import com.ubtrobot.navigation.NavigationManager
import com.ubtrobot.navigation.Location
import com.ubtrobot.speech.SpeechManager
import com.ubtrobot.navigation.Point

import org.eclipse.paho.client.mqttv3.MqttClient
import org.eclipse.paho.client.mqttv3.MqttConnectOptions
import org.eclipse.paho.client.mqttv3.MqttMessage
import org.eclipse.paho.client.mqttv3.MqttCallback
import org.eclipse.paho.client.mqttv3.IMqttDeliveryToken
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val httpClient = OkHttpClient()

    private var currentSessionId: String? = null
    private val detectedCategories = mutableListOf<String>()
    private val currentRecommendations = mutableListOf<RecommendationItem>()
    private var currentConversationState: JSONObject? = null
    private var currentIncludeFilters: JSONObject? = null
    private var currentTab: String = "scan"

    private var mqttClient: MqttClient? = null

    // Hardware Managers (V2.8.0 Architecture)
    private var navigationManager: NavigationManager? = null
    private var speechManager: SpeechManager? = null
    private lateinit var textToSpeech: TextToSpeech

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
            uploadScan(bitmap)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize Hardware
        initCruzrHardware()

        // START THE NETWORK BRIDGE
        startMqttListener()

        setStatus("Ready.")
        updateSessionLabel()
        renderDetections()
        renderRecommendations()
        switchTab("scan")

        textToSpeech = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                // Optional: Force English (or use Locale("pt", "BR") for Portuguese)
                textToSpeech.language = java.util.Locale.US
                android.util.Log.d("CruzrApp", "TTS Ready!")
            }
        }

        binding.captureButton.setOnClickListener {
            launchCamera()
        }

        binding.sendChatButton.setOnClickListener {
            sendChat()
        }

        binding.recommendationsLeftButton.setOnClickListener {
            scrollRecommendations(-1)
        }

        binding.recommendationsRightButton.setOnClickListener {
            scrollRecommendations(1)
        }

        binding.connectionSettingsButton.setOnClickListener {
            showConnectionSettingsDialog()
        }

        binding.tabScanButton.setOnClickListener {
            switchTab("scan")
        }

        binding.tabRefineButton.setOnClickListener {
            switchTab("refine")
        }
    }

    private fun launchCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            cameraLauncher.launch(null)
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun uploadScan(bitmap: Bitmap) {
        setStatus("Uploading scan...")
        val baseUrl = normalizedBaseUrl() ?: return
        val jpegBytes = bitmap.toJpegBytes()

        val imageBody = jpegBytes.toRequestBody("image/jpeg".toMediaType())
        val multipartBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("file", "scan.jpg", imageBody)
            .build()

        val request = Request.Builder()
            .url("$baseUrl/api/mobile/scan")
            .post(multipartBody)
            .build()

        Thread {
            try {
                httpClient.newCall(request).execute().use { response ->
                    val bodyText = response.body?.string().orEmpty()
                    if (!response.isSuccessful) {
                        runOnUiThread {
                            setStatus("Scan failed: HTTP ${response.code}")
                            binding.chatReplyText.text = bodyText.ifBlank { "No error body returned." }
                        }
                        return@use
                    }

                    val json = JSONObject(bodyText)
                    currentSessionId = json.optString("session_id").ifBlank { null }
                    currentConversationState = null
                    currentIncludeFilters = null
                    updateDetections(json.optJSONArray("detections"))
                    updateRecommendations(parseRecommendations(json.optJSONArray("recommendations")))
                    updateAnnotatedImage(json.optString("annotated_frame"))

                    runOnUiThread {
                        switchTab("scan")
                        updateSessionLabel("Vision-led")
                        binding.chatReplyText.text = "Scan complete. Tap a recommendation to inspect it, or refine with chat."
                        setStatus("Scan complete.")
                    }
                }
            } catch (exc: Exception) {
                runOnUiThread {
                    setStatus("Scan request failed.")
                    binding.chatReplyText.text = exc.message ?: "Unknown error"
                }
            }
        }.start()
    }

    private fun sendChat() {
        val message = binding.chatInput.text.toString().trim()
        if (message.isBlank()) {
            toast("Enter a refinement message first.")
            return
        }

        val baseUrl = normalizedBaseUrl() ?: return
        setStatus("Sending refinement...")

        val payload = JSONObject().apply {
            put("message", message)
            put("session_id", currentSessionId)
            put("replace_vision", binding.replaceVisionSwitch.isChecked)
            put("detected_categories", JSONArray(detectedCategories))
            put("history", JSONArray())
            put("recommendations", JSONArray(currentRecommendations.map { it.toJson() }))
            currentConversationState?.let { put("state", it) }
        }

        val request = Request.Builder()
            .url("$baseUrl/api/chat")
            .post(payload.toString().toRequestBody("application/json".toMediaType()))
            .build()

        Thread {
            try {
                httpClient.newCall(request).execute().use { response ->
                    val bodyText = response.body?.string().orEmpty()
                    if (!response.isSuccessful) {
                        runOnUiThread {
                            setStatus("Chat failed: HTTP ${response.code}")
                            binding.chatReplyText.text = extractErrorMessage(bodyText)
                        }
                        return@use
                    }

                    val json = JSONObject(bodyText)
                    currentSessionId = json.optString("session_id").ifBlank { currentSessionId }
                    currentConversationState = json.optJSONObject("state")
                    currentIncludeFilters = extractIncludeFilters(json)
                    updateRecommendations(parseRecommendations(json.optJSONArray("results")))

                    runOnUiThread {
                        switchTab("refine")
                        val mode = if (binding.replaceVisionSwitch.isChecked) "Search-led override" else "Vision + search"
                        updateSessionLabel(mode)
                        binding.chatReplyText.text = json.optString("reply", "No reply returned.")
                        setStatus("Refinement complete.")
                        binding.chatInput.text?.clear()
                    }
                }
            } catch (exc: Exception) {
                runOnUiThread {
                    setStatus("Chat request failed.")
                    binding.chatReplyText.text = exc.message ?: "Unknown error"
                }
            }
        }.start()
    }

    private fun updateDetections(detections: JSONArray?) {
        detectedCategories.clear()
        if (detections != null) {
            for (i in 0 until detections.length()) {
                val det = detections.optJSONObject(i) ?: continue
                val name = det.optString("class_name", "").trim()
                if (name.isNotBlank()) {
                    detectedCategories.add(name)
                }
            }
        }
        runOnUiThread { renderDetections() }
    }

    private fun renderDetections() {
        binding.detectionsGroup.removeAllViews()
        if (detectedCategories.isEmpty()) {
            val chip = buildDetectionChip(getString(R.string.detections_empty))
            binding.detectionsGroup.addView(chip)
            return
        }

        detectedCategories.distinct().forEach { category ->
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

    private fun updateRecommendations(items: List<RecommendationItem>) {
        currentRecommendations.clear()
        currentRecommendations.addAll(items)
        runOnUiThread { renderRecommendations() }
    }

    private fun renderRecommendations() {
        binding.recommendationsStrip.removeAllViews()
        val hasItems = currentRecommendations.isNotEmpty()
        binding.recommendationsEmptyText.isVisible = !hasItems
        binding.recommendationsScroll.isVisible = hasItems
        binding.recommendationsLeftButton.isEnabled = hasItems
        binding.recommendationsRightButton.isEnabled = hasItems

        if (!hasItems) return

        currentRecommendations.forEachIndexed { index, item ->
            binding.recommendationsStrip.addView(buildRecommendationCard(item, index))
        }
        binding.recommendationsScroll.post {
            binding.recommendationsScroll.scrollTo(0, 0)
        }
    }

    private fun buildRecommendationCard(item: RecommendationItem, index: Int): View {
        val card = MaterialCardView(this).apply {
            layoutParams = LinearLayout.LayoutParams(dp(220), LinearLayout.LayoutParams.MATCH_PARENT).also {
                it.marginEnd = dp(12)
            }
            radius = dp(22).toFloat()
            strokeWidth = dp(1)
            setStrokeColor(ContextCompat.getColor(context, R.color.brand_border))
            cardElevation = 0f
            setCardBackgroundColor(ContextCompat.getColor(context, R.color.white))
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
                foreground = ContextCompat.getDrawable(context, android.R.drawable.list_selector_background)
            }
            isClickable = true
            isFocusable = true
            setOnClickListener { showRecommendationDetail(item) }
        }

        val content = LinearLayout(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT
            )
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(16), dp(16), dp(16))
        }

        content.addView(TextView(this).apply {
            text = item.name
            setTextColor(ContextCompat.getColor(context, R.color.brand_text))
            setTypeface(typeface, Typeface.BOLD)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
        })

        content.addView(TextView(this).apply {
            text = item.category.replace("_", " ").uppercase()
            setTextColor(ContextCompat.getColor(context, R.color.brand_muted))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
            setTypeface(typeface, Typeface.BOLD)
            setPadding(0, dp(6), 0, 0)
        })

        content.addView(TextView(this).apply {
            text = item.reason.ifBlank { "Recommended from your search context." }
            setTextColor(ContextCompat.getColor(context, R.color.brand_muted))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
            setPadding(0, dp(10), 0, 0)
        })

        content.addView(TextView(this).apply {
            text = item.price
            setTextColor(ContextCompat.getColor(context, R.color.brand_accent_strong))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
            setTypeface(typeface, Typeface.BOLD)
            setPadding(0, dp(14), 0, 0)
        })

        if (index == currentRecommendations.lastIndex) {
            (card.layoutParams as LinearLayout.LayoutParams).marginEnd = 0
        }

        card.addView(content)
        return card
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
                // Natively command the robot instead of making a network call
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
        val direct = extractIncludeFilters(currentIncludeFilters)
        if (direct != null && direct.length() > 0) return direct

        val stateFilters = currentConversationState?.optJSONObject("filters")
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

    private fun parseRecommendations(items: JSONArray?): List<RecommendationItem> {
        if (items == null) return emptyList()
        return buildList {
            for (i in 0 until items.length()) {
                val item = items.optJSONObject(i) ?: continue
                add(RecommendationItem.fromJson(item))
            }
        }
    }

    private fun updateAnnotatedImage(base64Image: String?) {
        if (base64Image.isNullOrBlank()) {
            return
        }
        try {
            val bytes = Base64.decode(base64Image, Base64.DEFAULT)
            val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            runOnUiThread {
                binding.resultImage.setImageBitmap(bitmap)
            }
        } catch (_: IllegalArgumentException) {
            runOnUiThread {
                binding.resultImage.setImageBitmap(null)
            }
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
        binding.statusText.text = "Status: $message"
    }

    private fun updateSessionLabel(mode: String? = null) {
        val sessionId = currentSessionId
        binding.sessionText.text = if (sessionId.isNullOrBlank()) {
            getString(R.string.session_waiting)
        } else {
            val modeSuffix = if (mode.isNullOrBlank()) "" else " | $mode"
            "Session: ${sessionId.take(8)}$modeSuffix"
        }
    }

    private fun extractErrorMessage(bodyText: String): String {
        return try {
            val json = JSONObject(bodyText)
            json.optString("detail").ifBlank { bodyText.ifBlank { "No error body returned." } }
        } catch (_: Exception) {
            bodyText.ifBlank { "No error body returned." }
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

    private fun Bitmap.toJpegBytes(): ByteArray {
        val stream = ByteArrayOutputStream()
        compress(Bitmap.CompressFormat.JPEG, 90, stream)
        return stream.toByteArray()
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

    private fun styleTabButton(button: MaterialButton, selected: Boolean) {
        if (selected) {
            button.setBackgroundColor(ContextCompat.getColor(this, R.color.brand_text))
            button.setTextColor(ContextCompat.getColor(this, R.color.white))
            button.strokeWidth = 0
        } else {
            button.setBackgroundColor(ContextCompat.getColor(this, R.color.brand_surface))
            button.setTextColor(ContextCompat.getColor(this, R.color.brand_text))
            button.strokeWidth = dp(1)
            button.setStrokeColorResource(R.color.brand_border)
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

    private fun startMqttListener() {
        // 1. Extract the raw IP address from your existing server URL string
        val rawUrl = loadServerUrl().replace("http://", "").replace("https://", "")
        val ipAddress = rawUrl.split(":")[0]

        val brokerUri = "tcp://$ipAddress:1883"
        val clientId = "Cruzr_Kotlin_App"

        try {
            mqttClient = MqttClient(brokerUri, clientId, MemoryPersistence())
            val options = MqttConnectOptions().apply {
                isCleanSession = true
                connectionTimeout = 10
            }

            // 2. Define what happens when a network message arrives
            mqttClient?.setCallback(object : MqttCallback {
                override fun connectionLost(cause: Throwable?) {
                    runOnUiThread { setStatus("MQTT Connection Lost: ${cause?.message}") }
                }

                override fun messageArrived(topic: String?, message: MqttMessage?) {
                    val payloadString = message?.toString() ?: return

                    // When a message arrives, parse the JSON
                    try {
                        val json = JSONObject(payloadString)
                        val action = json.optString("action")

                        // If the PC Python script says "move_to_stand"...
                        if (action == "move_to_stand") {
                            val targetRack = json.optString("target", "unknown_rack")

                            runOnUiThread {
                                toast("Received PC Command: Go to $targetRack!")
                            }

                            // Trigger your existing hardware function!
                            sendRobotNavigationCommand(targetRack, "AI recommendation received via network")
                        }
                        // NEW LOGIC: If the script says "speak"...
                        else if (action == "speak") {
                            val textToSay = json.optString("text", "Meow meow")

                            runOnUiThread {
                                setStatus("Speaking: $textToSay")
                            }

                            // Trigger the text-to-speech engine
                            textToSpeech.speak(textToSay, TextToSpeech.QUEUE_FLUSH, null, null)
                        }
                    } catch (e: Exception) {
                        e.printStackTrace()
                    }
                }

                override fun deliveryComplete(token: IMqttDeliveryToken?) {}
            })

            // 3. Connect and Subscribe to the topic
            mqttClient?.connect(options)
            mqttClient?.subscribe("cruzr/commands")
            runOnUiThread { setStatus("MQTT Connected to $ipAddress") }

        } catch (e: Exception) {
            runOnUiThread { setStatus("MQTT Setup Failed: ${e.message}") }
        }
    }

    // --- HARDWARE INTEGRATION METHODS ---
    private fun initCruzrHardware() {
        setStatus("Initializing V2.8.0 Hardware Managers...")

        try {
            // Fetch the active managers from the Robot's internal OS context
            navigationManager = Robot.globalContext().getSystemService(NavigationManager.SERVICE) as NavigationManager

            // SpeechManager usually uses a similar constant or the string "speech"
            speechManager = Robot.globalContext().getSystemService("speech") as SpeechManager

            runOnUiThread { setStatus("Hardware Managers active.") }
        } catch (e: Exception) {
            runOnUiThread { setStatus("Manager Init Failed: ${e.message}") }
        }
    }

    private fun sendRobotNavigationCommand(targetItem: String, reason: String) {
        runOnUiThread { setStatus("Commanding hardware...") }

        val textToSpeak = if (reason.isNotBlank()) {
            "I found a great match! $reason. Let me show you where it is."
        } else {
            "Let me show you where the $targetItem is."
        }

        try {
            // 1. Trigger Speech
            speechManager?.synthesize(textToSpeak)

            // 2. Fetch the current map to find the destination
            navigationManager?.currentNavMap?.done { navMap ->

                // Get the list of markers using the exact method from the decompiled NavMap class
                val markers = navMap.markerList

                if (markers != null) {
                    // Search the list using the Marker's TITLE property
                    val targetMarker = markers.find { it.title.equals(targetItem, ignoreCase = true) }

                    if (targetMarker != null) {
                        // Because Marker extends Location, we pass it directly to navigate!
                        navigationManager?.navigate(targetMarker)

                        runOnUiThread {
                            toast("Cruzr is moving to $targetItem!")
                            setStatus("Navigating to location...")
                        }
                    } else {
                        runOnUiThread {
                            toast("'$targetItem' is not mapped on the robot.")
                            setStatus("Target missing from map.")
                        }
                    }
                }

            }?.fail {
                runOnUiThread {
                    toast("Could not access the robot's map.")
                    setStatus("Map access failed.")
                }
            }

        } catch (exc: Exception) {
            runOnUiThread {
                toast("Hardware execution failure: ${exc.message}")
                setStatus("Manager failed.")
            }
        }
    }

    override fun onDestroy() {
        if (::textToSpeech.isInitialized) {
            textToSpeech.stop()
            textToSpeech.shutdown()
        }
        super.onDestroy()
    }
}