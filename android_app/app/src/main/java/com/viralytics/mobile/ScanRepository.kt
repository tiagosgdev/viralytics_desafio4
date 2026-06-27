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
    val detectionLabels: List<String>,     // "beige trousers" format for display
    val recommendations: List<MainActivity.RecommendationItem>,
    val annotatedFrameBase64: String?,
    val bodyAnnotatedFrameBase64: String?, // skeleton pose overlay
    val bodyShape: String?,
)

class ScanRepository(private val httpClient: OkHttpClient) {

    suspend fun scan(
        bitmap: Bitmap,
        baseUrl: String,
        persona: String,
        userGender: String = "",
        userHeightCm: Int = 0,
    ): Result<ScanResult> =
        withContext(Dispatchers.IO) {
            runCatching {
                val jpegBytes = bitmap.toJpegBytes()
                val imageBody = jpegBytes.toRequestBody("image/jpeg".toMediaType())
                val builder = MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("file", "scan.jpg", imageBody)
                    .addFormDataPart("persona", persona)
                    .addFormDataPart("gender", userGender)
                if (userHeightCm > 0) builder.addFormDataPart("height_cm", userHeightCm.toString())
                val multipartBody = builder.build()

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
                    val detArray = json.optJSONArray("detections")
                    val bodyAnalysis = json.optJSONObject("body_analysis")
                    ScanResult(
                        sessionId = json.optString("session_id").ifBlank { null },
                        detections = parseDetectionNames(detArray),
                        detectionLabels = parseDetectionLabels(detArray),
                        recommendations = parseRecommendations(json.optJSONArray("recommendations")),
                        annotatedFrameBase64 = json.optString("annotated_frame").ifBlank { null },
                        bodyAnnotatedFrameBase64 = json.optString("body_annotated_frame").ifBlank { null },
                        bodyShape = bodyAnalysis?.optString("body_shape")
                            ?.takeIf { it.isNotBlank() && it != "unknown" },
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

    private fun parseDetectionLabels(array: org.json.JSONArray?): List<String> {
        if (array == null) return emptyList()
        return buildList {
            for (i in 0 until array.length()) {
                val obj = array.optJSONObject(i) ?: continue
                val name = obj.optString("class_name").trim().takeIf { it.isNotBlank() } ?: continue
                val color = obj.optString("color_name").trim()
                val display = name.replace('_', ' ')
                add(if (color.isNotBlank()) "$color $display" else display)
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

    private fun Bitmap.toJpegBytes(maxDim: Int = 1280): ByteArray {
        val scaled = if (width > maxDim || height > maxDim) {
            val scale = maxDim.toFloat() / maxOf(width, height)
            Bitmap.createScaledBitmap(this, (width * scale).toInt(), (height * scale).toInt(), true)
        } else this
        val stream = ByteArrayOutputStream()
        scaled.compress(Bitmap.CompressFormat.JPEG, 85, stream)
        if (scaled !== this) scaled.recycle()
        return stream.toByteArray()
    }
}
