package com.viralytics.mobile

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

class AgentRepository(private val httpClient: OkHttpClient) {

    suspend fun recommend(
        baseUrl: String,
        detectedType: String,
        detectedBodyType: String,
        detectedColor: String,
        userAnswer: String,
        userGender: String = "",
    ): Result<Pair<String?, List<MainActivity.RecommendationItem>>> = withContext(Dispatchers.IO) {
        runCatching {
            val payload = JSONObject().apply {
                put("detected_type", detectedType)
                put("detected_body_type", detectedBodyType)
                put("detected_color", detectedColor)
                put("user_gender", userGender)
                put("user_answer", userAnswer)
            }
            val request = Request.Builder()
                .url("$baseUrl/api/recommend")
                .post(payload.toString().toRequestBody("application/json".toMediaType()))
                .build()

            httpClient.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return@runCatching null to emptyList()
                val json = JSONObject(response.body?.string().orEmpty())
                val roundId = json.optString("round_id").takeIf { it.isNotBlank() }
                val arr = json.optJSONArray("recommendations") ?: return@runCatching roundId to emptyList()
                val items = buildList {
                    for (i in 0 until arr.length()) {
                        val item = arr.optJSONObject(i) ?: continue
                        add(MainActivity.RecommendationItem.fromAgentJson(item))
                    }
                }
                roundId to items
            }
        }
    }
}
