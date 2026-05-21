import type { PredictionResponse } from "@/types/prediction";

export async function predictScoliosisImage(
  image: File,
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", image);

  let response: Response;

  try {
    response = await fetch("/api/segment-rbunet", {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Unable to reach the prediction API.");
  }

  let payload: unknown = null;

  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    const detail =
      payload && typeof payload === "object" && "detail" in payload
        ? String(payload.detail)
        : "Prediction request failed.";

    throw new Error(detail);
  }

  if (!payload || typeof payload !== "object") {
    return { message: "Prediction completed, but no structured data returned." };
  }

  return payload as PredictionResponse;
}
