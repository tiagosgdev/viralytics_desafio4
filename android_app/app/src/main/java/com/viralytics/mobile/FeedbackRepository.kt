package com.viralytics.mobile

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

class FeedbackRepository(private val httpClient: OkHttpClient) {

    /**
     * Best-effort POST to /api/feedback. The server never raises — it returns
     * {ok, applied, reason, policy} (ok=False when the round/item is unknown).
     * Callers must treat ok=False as informational, not an error.
     */
    suspend fun submit(
        baseUrl: String,
        roundId: String,
        itemId: Int,
        size: String,
        rating: Int,
    ): Result<JSONObject> = withContext(Dispatchers.IO) {
        runCatching {
            val payload = JSONObject().apply {
                put("round_id", roundId)
                put("item_id", itemId)
                put("size", size)
                put("rating", rating)
            }
            val request = Request.Builder()
                .url("$baseUrl/api/feedback")
                .post(payload.toString().toRequestBody("application/json".toMediaType()))
                .build()

            httpClient.newCall(request).execute().use { response ->
                JSONObject(response.body?.string().orEmpty())
            }
        }
    }
}
