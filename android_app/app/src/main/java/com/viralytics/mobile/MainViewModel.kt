package com.viralytics.mobile

import android.graphics.Bitmap
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

data class HistoryEntry(val role: String, val content: String)

sealed interface UiEvent {
    data class ShowToast(val message: String) : UiEvent
    data class SetStatus(val message: String) : UiEvent
    data class ScanComplete(
        val sessionId: String?,
        val detections: List<String>,
        val detectionLabels: List<String>,
        val recommendations: List<MainActivity.RecommendationItem>,
        val annotatedFrameBase64: String?,
        val bodyAnnotatedFrameBase64: String?,
        val bodyShape: String?,
    ) : UiEvent
    data class ChatComplete(
        val sessionId: String?,
        val reply: String,
        val recommendations: List<MainActivity.RecommendationItem>,
    ) : UiEvent
    data class AgentRecsComplete(
        val recommendations: List<MainActivity.RecommendationItem>,
    ) : UiEvent
    data class ScanError(val message: String) : UiEvent
    data class ChatError(val message: String) : UiEvent
}

class MainViewModel : ViewModel() {

    private val httpClient = run {
        val trustAll = arrayOf<javax.net.ssl.TrustManager>(object : javax.net.ssl.X509TrustManager {
            override fun checkClientTrusted(chain: Array<java.security.cert.X509Certificate>, authType: String) {}
            override fun checkServerTrusted(chain: Array<java.security.cert.X509Certificate>, authType: String) {}
            override fun getAcceptedIssuers(): Array<java.security.cert.X509Certificate> = arrayOf()
        })
        val sslCtx = javax.net.ssl.SSLContext.getInstance("TLS")
        sslCtx.init(null, trustAll, java.security.SecureRandom())
        OkHttpClient.Builder()
            .connectTimeout(15, TimeUnit.SECONDS)
            .readTimeout(90, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .sslSocketFactory(sslCtx.socketFactory, trustAll[0] as javax.net.ssl.X509TrustManager)
            .hostnameVerifier { _, _ -> true }
            .build()
    }
    private val scanRepository = ScanRepository(httpClient)
    private val chatRepository = ChatRepository(httpClient)
    private val sessionRepository = SessionRepository(httpClient)
    private val agentRepository = AgentRepository(httpClient)
    private val feedbackRepository = FeedbackRepository(httpClient)

    private val _events = MutableLiveData<UiEvent>()
    val events: LiveData<UiEvent> = _events

    // Persona persisted across the session
    var selectedPersona: String = "cruella"

    // State that survives rotation
    var currentSessionId: String? = null
    val detectedCategories = mutableListOf<String>()
    val detectionLabels = mutableListOf<String>()
    val currentRecommendations = mutableListOf<MainActivity.RecommendationItem>()
    var currentConversationState: org.json.JSONObject? = null
    var currentIncludeFilters: org.json.JSONObject? = null
    val chatHistory = mutableListOf<HistoryEntry>()

    // Agent-recommendation parity state
    var detectedType: String = ""
    var detectedColor: String = ""
    var detectedBodyType: String = ""
    var selectedGender: String = ""
    var selectedHeightCm: Int = 0
    var currentRoundId: String? = null
    private val searchIntentMessages = mutableListOf<String>()

    fun uploadScan(bitmap: Bitmap, baseUrl: String, persona: String) {
        selectedPersona = persona
        viewModelScope.launch {
            _events.value = UiEvent.SetStatus("Uploading scan...")
            scanRepository.scan(bitmap, baseUrl, persona, userGender = selectedGender, userHeightCm = selectedHeightCm)
                .onSuccess { result ->
                    currentSessionId = result.sessionId
                    currentConversationState = null
                    currentIncludeFilters = null
                    chatHistory.clear()
                    detectedCategories.clear()
                    detectedCategories.addAll(result.detections)
                    detectionLabels.clear()
                    detectionLabels.addAll(result.detectionLabels)
                    currentRecommendations.clear()
                    currentRecommendations.addAll(result.recommendations)
                    _events.value = UiEvent.ScanComplete(
                        sessionId = result.sessionId,
                        detections = result.detections,
                        detectionLabels = result.detectionLabels,
                        recommendations = result.recommendations,
                        annotatedFrameBase64 = result.annotatedFrameBase64,
                        bodyAnnotatedFrameBase64 = result.bodyAnnotatedFrameBase64,
                        bodyShape = result.bodyShape,
                    )
                    _events.value = UiEvent.SetStatus("Scan complete.")
                    viewModelScope.launch { startSession(baseUrl, persona, result.detections, result.recommendations) }
                    // NOTE: agent recommendations are fetched on the TABLET (via the MQTT
                    // injectScanResult path), not here on the camera-only phone — the phone
                    // never displays recs, so fetching them here was wasted work.
                }
                .onFailure { err ->
                    _events.value = UiEvent.ScanError(err.message ?: "Unknown scan error")
                    _events.value = UiEvent.SetStatus("Scan request failed.")
                }
        }
    }

    private suspend fun startSession(
        baseUrl: String,
        persona: String,
        detections: List<String>,
        recommendations: List<MainActivity.RecommendationItem>,
    ) {
        sessionRepository.startSession(baseUrl, persona, detections, recommendations)
            .onSuccess { sessionId ->
                if (sessionId != null) currentSessionId = sessionId
            }
    }

    private suspend fun fetchAgentRecommendations(
        baseUrl: String,
        type: String,
        color: String,
        bodyType: String,
        userAnswer: String,
    ) {
        _events.value = UiEvent.SetStatus("Agents computing…")
        agentRepository.recommend(
            baseUrl = baseUrl,
            detectedType = type,
            detectedBodyType = bodyType,
            detectedColor = color,
            userAnswer = userAnswer,
            userGender = selectedGender,
        )
            .onSuccess { (roundId, recs) ->
                if (recs.isNotEmpty()) {
                    currentRoundId = roundId
                    currentRecommendations.clear()
                    currentRecommendations.addAll(recs)
                    _events.value = UiEvent.AgentRecsComplete(recs)
                    _events.value = UiEvent.SetStatus("Agent recommendations ready.")
                } else {
                    _events.value = UiEvent.SetStatus("Scan complete.")
                }
            }
            .onFailure {
                _events.value = UiEvent.SetStatus("Scan complete.")
            }
    }

    fun sendChat(
        message: String,
        baseUrl: String,
        replaceVision: Boolean,
        persona: String,
    ) {
        selectedPersona = persona
        chatHistory.add(HistoryEntry("user", message))
        searchIntentMessages.add(message)
        viewModelScope.launch {
            _events.value = UiEvent.SetStatus("Sending refinement...")
            chatRepository.sendMessage(
                message = message,
                baseUrl = baseUrl,
                sessionId = currentSessionId,
                persona = persona,
                replaceVision = replaceVision,
                detectedCategories = detectedCategories.toList(),
                currentRecommendations = currentRecommendations.toList(),
                conversationState = currentConversationState,
                history = chatHistory.toList(),
            )
                .onSuccess { result ->
                    currentSessionId = result.sessionId
                    currentConversationState = result.conversationState
                    currentIncludeFilters = result.includeFilters
                    if (result.recommendations.isNotEmpty()) {
                        currentRecommendations.clear()
                        currentRecommendations.addAll(result.recommendations)
                    }
                    chatHistory.add(HistoryEntry("assistant", result.reply))
                    _events.value = UiEvent.ChatComplete(
                        sessionId = result.sessionId,
                        reply = result.reply,
                        recommendations = result.recommendations,
                    )
                    _events.value = UiEvent.SetStatus("Refinement complete.")
                    maybeTriggerAgentRound(result, baseUrl)
                }
                .onFailure { err ->
                    if (chatHistory.isNotEmpty()) chatHistory.removeAt(chatHistory.lastIndex)
                    _events.value = UiEvent.ChatError(err.message ?: "Unknown chat error")
                    _events.value = UiEvent.SetStatus("Chat request failed.")
                }
        }
    }

    fun injectScanResult(
        sessionId: String?,
        detections: List<String>,
        detectionLabels: List<String>,
        recommendations: List<MainActivity.RecommendationItem>,
        annotatedFrameBase64: String?,
        bodyAnnotatedFrameBase64: String?,
        bodyShape: String?,
        detectedColor: String,
        detectedBodyType: String,
        baseUrl: String?,
    ) {
        currentSessionId = sessionId
        currentConversationState = null
        currentIncludeFilters = null
        chatHistory.clear()
        detectedCategories.clear()
        detectedCategories.addAll(detections)
        this.detectionLabels.clear()
        this.detectionLabels.addAll(detectionLabels)
        currentRecommendations.clear()
        currentRecommendations.addAll(recommendations)

        // Reset agent-rec parity state for the new scan.
        this.detectedType = detections.firstOrNull().orEmpty()
        this.detectedColor = detectedColor
        this.detectedBodyType = detectedBodyType
        currentRoundId = null
        searchIntentMessages.clear()

        _events.postValue(UiEvent.ScanComplete(
            sessionId = sessionId,
            detections = detections,
            detectionLabels = detectionLabels,
            recommendations = recommendations,
            annotatedFrameBase64 = annotatedFrameBase64,
            bodyAnnotatedFrameBase64 = bodyAnnotatedFrameBase64,
            bodyShape = bodyShape,
        ))

        // Tablet fetches agent recommendations over HTTP using the scan signals.
        if (baseUrl != null) {
            viewModelScope.launch {
                fetchAgentRecommendations(
                    baseUrl = baseUrl,
                    type = this@MainViewModel.detectedType,
                    color = this@MainViewModel.detectedColor,
                    bodyType = this@MainViewModel.detectedBodyType,
                    userAnswer = "",
                )
            }
        }
    }

    fun navigateByCategory(baseUrl: String, category: String) {
        viewModelScope.launch {
            _events.value = UiEvent.SetStatus("Sending navigation request…")
            runCatching {
                withContext(Dispatchers.IO) {
                    val body = JSONObject().put("category", category).toString()
                        .toRequestBody("application/json".toMediaType())
                    val request = Request.Builder()
                        .url("$baseUrl/api/robot/navigate-by-category")
                        .post(body)
                        .build()
                    httpClient.newCall(request).execute().use { response ->
                        if (!response.isSuccessful) {
                            val msg = response.body?.string() ?: "HTTP ${response.code}"
                            throw Exception(msg)
                        }
                    }
                }
            }.onFailure { e ->
                _events.value = UiEvent.SetStatus("Navigation request failed: ${e.message}")
            }
        }
    }

    fun clearSession() {
        currentSessionId = null
        currentConversationState = null
        currentIncludeFilters = null
        chatHistory.clear()
        detectedCategories.clear()
        detectionLabels.clear()
        currentRecommendations.clear()
        currentRoundId = null
        searchIntentMessages.clear()
        selectedGender = ""
        selectedHeightCm = 0
    }

    /**
     * Escalates to a styled agent round when the chat reports a completed search.
     * Mirrors web `maybeTriggerAgentRound` (frontend/js/ui/chat.js:122-147).
     */
    private suspend fun maybeTriggerAgentRound(result: ChatResult, baseUrl: String) {
        if (result.action != "searched") return
        val include = extractInclude(result.activeFilters)
        val briefType = include?.optJSONArray("type")?.optString(0)?.takeIf { it.isNotBlank() }
        val briefColor = include?.optJSONArray("color")?.optString(0)?.takeIf { it.isNotBlank() }
        val type = briefType ?: detectedType
        val color = briefColor ?: detectedColor
        val intent = accumulatedUserIntent()
        searchIntentMessages.clear()
        fetchAgentRecommendations(
            baseUrl = baseUrl,
            type = type,
            color = color,
            bodyType = detectedBodyType,
            userAnswer = intent,
        )
    }

    /** Reads the nested `include` object from an `active_filters` payload. */
    private fun extractInclude(src: JSONObject?): JSONObject? = src?.optJSONObject("include")

    /**
     * Joins recent non-confirmation user messages into a single intent string.
     * Mirrors web `accumulatedUserIntent` (frontend/js/ui/chat.js:114-120).
     */
    private fun accumulatedUserIntent(): String =
        searchIntentMessages
            .map { it.trim() }
            .filter { it.isNotBlank() && !CONFIRM_WORDS.contains(it.lowercase()) }
            .takeLast(6)
            .joinToString(". ")

    /**
     * Best-effort feedback submit. Guards on round_id + item_id, posts off the main
     * thread, and reports the outcome via [onResult] on the main thread. ok=False is
     * never treated as an error.
     */
    fun submitFeedback(
        baseUrl: String?,
        item: MainActivity.RecommendationItem,
        rating: Int,
        onResult: (String) -> Unit,
    ) {
        val roundId = currentRoundId
        val itemId = item.itemId
        if (baseUrl == null || roundId == null || itemId == null) {
            onResult("Feedback unavailable for this item.")
            return
        }
        viewModelScope.launch {
            val message = feedbackRepository.submit(
                baseUrl = baseUrl,
                roundId = roundId,
                itemId = itemId,
                size = item.size ?: item.metadata["size"] ?: "",
                rating = rating,
            ).fold(
                onSuccess = { json ->
                    if (json.optBoolean("ok")) {
                        if (json.optBoolean("applied")) "Thanks — the agent is learning 🎓"
                        else "Thanks! Neutral — no change."
                    } else {
                        json.optString("reason").takeIf { it.isNotBlank() } ?: "Feedback saved."
                    }
                },
                onFailure = { "Could not send feedback." },
            )
            onResult(message)
        }
    }

    companion object {
        private val CONFIRM_WORDS = setOf(
            "yes", "y", "sure", "ok", "okay", "go ahead", "confirm", "i confirm",
            "correct", "correcy", "that's right", "thats right", "that is perfect",
            "that's perfect", "perfect", "looks good", "looks perfect", "sounds good",
            "works for me", "i like that", "please proceed", "proceed", "approved",
            "do it", "run it", "search now", "run the search", "why not",
        )
    }
}
