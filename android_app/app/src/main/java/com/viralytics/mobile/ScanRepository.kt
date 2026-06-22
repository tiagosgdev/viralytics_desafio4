package com.viralytics.mobile

import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream

data class ScanResult(
    val sessionId: String?,
    val detections: List<String>,
    val recommendations: List<MainActivity.RecommendationItem>,
    val annotatedFrameBase64: String?,
)

class ScanRepository(private val httpClient: OkHttpClient) {

    suspend fun scan(bitmap: Bitmap, baseUrl: String, persona: String): Result<ScanResult> =
        withContext(Dispatchers.IO) {
            runCatching {
                val jpegBytes = bitmap.toJpegBytes()
                val imageBody = jpegBytes.toRequestBody("image/jpeg".toMediaType())
                val multipartBody = MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("file", "scan.jpg", imageBody)
                    .addFormDataPart("persona", persona)
                    .build()

                val request = Request.Builder()
                    .url("$baseUrl/api/mobile/scan")
                    .post(multipartBody)
                    .build()

                httpClient.newCall(request).execute().use { response ->
                    val body = response.body?.string().orEmpty()
                    if (!response.isSuccessful) {
                        error("Scan failed: HTTP ${response.code} — $body")
                    }
                    val json = JSONObject(body)
                    ScanResult(
                        sessionId = json.optString("session_id").ifBlank { null },
                        detections = parseDetectionNames(json.optJSONArray("detections")),
                        recommendations = parseRecommendations(json.optJSONArray("recommendations")),
                        annotatedFrameBase64 = json.optString("annotated_frame").ifBlank { null },
                    )
                }
            }
        }

    private fun parseDetectionNames(array: org.json.JSONArray?): List<String> {
        if (array == null) return emptyList()
        return buildList {
            for (i in 0 until array.length()) {
                val name = array.optJSONObject(i)?.optString("class_name")?.trim() ?: continue
                if (name.isNotBlank()) add(name)
            }
        }
    }

    private fun parseRecommendations(array: org.json.JSONArray?): List<MainActivity.RecommendationItem> {
        if (array == null) return emptyList()
        return buildList {
            for (i in 0 until array.length()) {
                val item = array.optJSONObject(i) ?: continue
                add(MainActivity.RecommendationItem.fromJson(item))
            }
        }
    }

    private fun Bitmap.toJpegBytes(): ByteArray {
        val stream = ByteArrayOutputStream()
        compress(Bitmap.CompressFormat.JPEG, 90, stream)
        return stream.toByteArray()
    }
}
