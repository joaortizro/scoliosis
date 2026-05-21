import type { PredictionModelKey, PredictionResponse } from "@/types/prediction";

const PUBLIC_BACKEND_API_BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_API_BASE_URL?.replace(/\/$/, "") ?? "";

const MODEL_ENDPOINTS: Record<PredictionModelKey, string> = {
  binary: "/predict-binary",
  full: "/predict-full",
  multiclass: "/predict-multiclass",
};

export const PREDICTION_MODEL_OPTIONS: {
  key: PredictionModelKey;
  label: string;
  description: string;
}[] = [
  {
    key: "full",
    label: "Full",
    description: "Runs the combined backend response when available.",
  },
  {
    key: "multiclass",
    label: "Multiclass",
    description: "Returns vertebra labels and instance geometry.",
  },
  {
    key: "binary",
    label: "Binary",
    description: "Returns the binary spine/vertebra segmentation model.",
  },
];

export async function predictScoliosisImage(
  image: File,
  model: PredictionModelKey = "full",
): Promise<PredictionResponse> {
  let response: Response;

  try {
    const formData = new FormData();
    formData.append("file", image);

    response = await fetch(`/api/model-predict?model=${model}`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Unable to reach the prediction API.");
  }

  const responseContentType = response.headers.get("content-type") ?? "";

  if (
    response.status === 404 &&
    PUBLIC_BACKEND_API_BASE_URL &&
    !responseContentType.includes("application/json")
  ) {
    try {
      const formData = new FormData();
      formData.append("file", image);

      response = await fetch(
        `${PUBLIC_BACKEND_API_BASE_URL}${MODEL_ENDPOINTS[model]}`,
        {
          method: "POST",
          body: formData,
        },
      );
    } catch {
      throw new Error("Unable to reach the public prediction API.");
    }
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
    const backendStatus =
      payload && typeof payload === "object" && "backendStatus" in payload
        ? ` Backend status: ${String(payload.backendStatus)}.`
        : "";
    const backendEndpoint =
      payload && typeof payload === "object" && "backendEndpoint" in payload
        ? ` Endpoint: ${String(payload.backendEndpoint)}.`
        : "";

    throw new Error(`${detail}${backendStatus}${backendEndpoint}`);
  }

  if (!payload || typeof payload !== "object") {
    return { message: "Prediction completed, but no structured data returned." };
  }

  return payload as PredictionResponse;
}
