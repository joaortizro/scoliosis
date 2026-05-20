import { DEFAULT_API_BASE_URL } from "@/lib/constants";
import type { PredictionResponse } from "@/types/prediction";

const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ??
  DEFAULT_API_BASE_URL;

export async function predictScoliosisImage(
  image: File,
): Promise<PredictionResponse> {
  if (!apiBaseUrl) {
    throw new Error(
      "Missing NEXT_PUBLIC_API_BASE_URL. Add it to frontend/.env.local.",
    );
  }

  const formData = new FormData();
  formData.append("file", image);

  let response: Response;

  try {
    response = await fetch(`${apiBaseUrl}/segment/rbunet?return_image=true`, {
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
