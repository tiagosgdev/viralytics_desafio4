package com.viralytics.mobile

import android.graphics.Bitmap
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import java.util.concurrent.TimeUnit

data class HistoryEntry(val role: String, val content: String)

sealed interface UiEvent {
    data class ShowToast(val message: String) : UiEvent
    data class SetStatus(val message: String) : UiEvent
    data class ScanComplete(
        val sessionId: String?,
        val detections: List<String>,
        val recommendations: List<MainActivity.RecommendationItem>,
        val annotatedFrameBase64: String?,
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

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(90, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()
    private val scanRepository = ScanRepository(httpClient)
    private val chatRepository = ChatRepository(httpClient)
    private val sessionRepository = SessionRepository(httpClient)
    private val agentRepository = AgentRepository(httpClient)

    private val _events = MutableLiveData<UiEvent>()
    val events: LiveData<UiEvent> = _events

    // Persona persisted across the session
    var selectedPersona: String = "cruella"

    // State that survives rotation
    var currentSessionId: String? = null
    val detectedCategories = mutableListOf<String>()
    val currentRecommendations = mutableListOf<MainActivity.RecommendationItem>()
    var currentConversationState: org.json.JSONObject? = null
    var currentIncludeFilters: org.json.JSONObject? = null
    val chatHistory = mutableListOf<HistoryEntry>()

    fun uploadScan(bitmap: Bitmap, baseUrl: String, persona: String) {
        selectedPersona = persona
        viewModelScope.launch {
            _events.value = UiEvent.SetStatus("Uploading scan...")
            scanRepository.scan(bitmap, baseUrl, persona)
                .onSuccess { result ->
                    currentSessionId = result.sessionId
                    currentConversationState = null
                    currentIncludeFilters = null
                    chatHistory.clear()
                    detectedCategories.clear()
                    detectedCategories.addAll(result.detections)
                    currentRecommendations.clear()
                    currentRecommendations.addAll(result.recommendations)
                    _events.value = UiEvent.ScanComplete(
                        sessionId = result.sessionId,
                        detections = result.detections,
                        recommendations = result.recommendations,
                        annotatedFrameBase64 = result.annotatedFrameBase64,
                    )
                    _events.value = UiEvent.SetStatus("Scan complete.")
                    startSession(baseUrl, persona, result.detections, result.recommendations)
                    fetchAgentRecommendations(baseUrl, result.detections.firstOrNull() ?: "")
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

    private suspend fun fetchAgentRecommendations(baseUrl: String, detectedType: String) {
        _events.value = UiEvent.SetStatus("Agents computing…")
        agentRepository.recommend(
            baseUrl = baseUrl,
            detectedType = detectedType,
            detectedBodyType = "",
            detectedColor = "",
        )
            .onSuccess { recs ->
                if (recs.isNotEmpty()) {
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
        recommendations: List<MainActivity.RecommendationItem>,
        annotatedFrameBase64: String?,
    ) {
        currentSessionId = sessionId
        currentConversationState = null
        currentIncludeFilters = null
        chatHistory.clear()
        detectedCategories.clear()
        detectedCategories.addAll(detections)
        currentRecommendations.clear()
        currentRecommendations.addAll(recommendations)
        _events.postValue(UiEvent.ScanComplete(
            sessionId = sessionId,
            detections = detections,
            recommendations = recommendations,
            annotatedFrameBase64 = annotatedFrameBase64,
        ))
    }

    fun clearSession() {
        currentSessionId = null
        currentConversationState = null
        currentIncludeFilters = null
        chatHistory.clear()
        detectedCategories.clear()
        currentRecommendations.clear()
    }
}
