package com.viralytics.mobile

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject

data class ChatResult(
    val sessionId: String?,
    val reply: String,
    val recommendations: List<MainActivity.RecommendationItem>,
    val conversationState: JSONObject?,
    val includeFilters: JSONObject?,
    val action: String?,
    val activeFilters: JSONObject?,
)

class ChatRepository(private val httpClient: OkHttpClient) {

    suspend fun sendMessage(
        message: String,
        baseUrl: String,
        sessionId: String?,
        persona: String,
        replaceVision: Boolean,
        detectedCategories: List<String>,
        currentRecommendations: List<MainActivity.RecommendationItem>,
        conversationState: JSONObject?,
        history: List<HistoryEntry>,
    ): Result<ChatResult> =
        withContext(Dispatchers.IO) {
            runCatching {
                val historyArray = JSONArray().apply {
                    history.forEach { entry ->
                        put(JSONObject().apply {
                            put("role", entry.role)
                            put("content", entry.content)
                        })
                    }
                }
                val payload = JSONObject().apply {
                    put("message", message)
                    put("persona", persona)
                    put("assistant_mode", persona)
                    put("session_id", sessionId)
                    put("replace_vision", replaceVision)
                    put("detected_categories", JSONArray(detectedCategories))
                    put("history", historyArray)
                    put("recommendations", JSONArray(currentRecommendations.map { it.toJson() }))
                    conversationState?.let { put("state", it) }
                }

                val request = Request.Builder()
                    .url("$baseUrl/api/chat")
                    .post(payload.toString().toRequestBody("application/json".toMediaType()))
                    .build()

                httpClient.newCall(request).execute().use { response ->
                    val body = response.body?.string().orEmpty()
                    if (!response.isSuccessful) {
                        error("Chat failed: HTTP ${response.code} — ${extractErrorMessage(body)}")
                    }
                    val json = JSONObject(body)
                    ChatResult(
                        sessionId = json.optString("session_id").ifBlank { sessionId },
                        reply = json.optString("reply", "No reply returned."),
                        recommendations = parseRecommendations(json.optJSONArray("results")),
                        conversationState = json.optJSONObject("state"),
                        includeFilters = extractIncludeFilters(json),
                        action = json.optString("action").takeIf { it.isNotBlank() },
                        activeFilters = json.optJSONObject("active_filters"),
                    )
                }
            }
        }

    private fun parseRecommendations(array: JSONArray?): List<MainActivity.RecommendationItem> {
        if (array == null) return emptyList()
        return buildList {
            for (i in 0 until array.length()) {
                val item = array.optJSONObject(i) ?: continue
                add(MainActivity.RecommendationItem.fromJson(item))
            }
        }
    }

    private fun extractIncludeFilters(source: JSONObject?): JSONObject? {
        if (source == null) return null
        if (source.has("include")) return source.optJSONObject("include")
        val keys = source.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            if (source.optJSONArray(key) != null) return source
        }
        return null
    }

    private fun extractErrorMessage(body: String): String {
        return try {
            val json = JSONObject(body)
            json.optString("detail").ifBlank { body.ifBlank { "No error body returned." } }
        } catch (_: Exception) {
            body.ifBlank { "No error body returned." }
        }
    }
}
