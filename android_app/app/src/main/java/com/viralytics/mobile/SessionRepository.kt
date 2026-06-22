package com.viralytics.mobile

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject

class SessionRepository(private val httpClient: OkHttpClient) {

    suspend fun startSession(
        baseUrl: String,
        persona: String,
        detectedCategories: List<String>,
        recommendations: List<MainActivity.RecommendationItem>,
    ): Result<String?> = withContext(Dispatchers.IO) {
        runCatching {
            val payload = JSONObject().apply {
                put("persona", persona)
                put("detected_categories", JSONArray(detectedCategories))
                put("recommendations", JSONArray(recommendations.map { it.toJson() }))
            }
            val request = Request.Builder()
                .url("$baseUrl/api/session/start")
                .post(payload.toString().toRequestBody("application/json".toMediaType()))
                .build()

            httpClient.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return@runCatching null
                JSONObject(response.body?.string().orEmpty()).optString("session_id").ifBlank { null }
            }
        }
    }
}
