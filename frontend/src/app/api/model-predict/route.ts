import { NextResponse } from "next/server";
import type { PredictionModelKey } from "@/types/prediction";

const MODEL_ENDPOINTS: Record<PredictionModelKey, string> = {
  binary: "/predict-binary",
  full: "/predict-full",
  multiclass: "/predict-multiclass",
};

function isPredictionModelKey(value: string): value is PredictionModelKey {
  return value === "binary" || value === "full" || value === "multiclass";
}

function getBackendHeaders() {
  const token =
    process.env.BACKEND_API_AUTH_TOKEN ??
    process.env.HUGGINGFACE_API_TOKEN ??
    process.env.HF_TOKEN;

  return token ? { Authorization: `Bearer ${token}` } : undefined;
}

function getErrorDetail(body: string, contentType: string, endpoint: string) {
  if (contentType.includes("application/json")) {
    try {
      const payload = JSON.parse(body) as unknown;

      if (payload && typeof payload === "object" && "detail" in payload) {
        return String(payload.detail);
      }
    } catch {
      // Fall through to the generic route-aware message.
    }
  }

  return `Prediction API returned ${endpoint} with a non-success response.`;
}

export async function POST(request: Request) {
  const apiBaseUrl = process.env.BACKEND_API_BASE_URL?.replace(/\/$/, "");

  if (!apiBaseUrl) {
    return NextResponse.json(
      { detail: "Missing BACKEND_API_BASE_URL environment variable." },
      { status: 500 },
    );
  }

  const url = new URL(request.url);
  const requestedModel = url.searchParams.get("model") ?? "full";

  if (!isPredictionModelKey(requestedModel)) {
    return NextResponse.json(
      { detail: "Unsupported prediction model endpoint." },
      { status: 400 },
    );
  }

  const endpoint = MODEL_ENDPOINTS[requestedModel];
  const formData = await request.formData();
  const response = await fetch(`${apiBaseUrl}${endpoint}`, {
    body: formData,
    headers: getBackendHeaders(),
    method: "POST",
  }).catch(() => null);

  if (!response) {
    return NextResponse.json(
      { detail: `Unable to reach the prediction API endpoint ${endpoint}.` },
      { status: 502 },
    );
  }

  const contentType = response.headers.get("content-type") ?? "application/json";
  const body = await response.text();

  if (!response.ok && !contentType.includes("application/json")) {
    return NextResponse.json(
      {
        detail: getErrorDetail(body, contentType, endpoint),
        backendEndpoint: endpoint,
        backendStatus: response.status,
      },
      { status: response.status },
    );
  }

  return new Response(body, {
    headers: { "content-type": contentType },
    status: response.status,
  });
}
