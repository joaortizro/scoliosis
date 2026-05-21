import { NextResponse } from "next/server";

function getBackendHeaders() {
  const token =
    process.env.BACKEND_API_AUTH_TOKEN ??
    process.env.HUGGINGFACE_API_TOKEN ??
    process.env.HF_TOKEN;

  return token ? { Authorization: `Bearer ${token}` } : undefined;
}

export async function POST(request: Request) {
  const apiBaseUrl = process.env.BACKEND_API_BASE_URL?.replace(/\/$/, "");

  if (!apiBaseUrl) {
    return NextResponse.json(
      { detail: "Missing BACKEND_API_BASE_URL environment variable." },
      { status: 500 },
    );
  }

  const formData = await request.formData();
  const response = await fetch(
    `${apiBaseUrl}/segment/rbunet?return_image=true`,
    {
      body: formData,
      headers: getBackendHeaders(),
      method: "POST",
    },
  ).catch(() => null);

  if (!response) {
    return NextResponse.json(
      { detail: "Unable to reach the segmentation API." },
      { status: 502 },
    );
  }

  const contentType = response.headers.get("content-type") ?? "application/json";
  const body = await response.text();

  return new Response(body, {
    headers: { "content-type": contentType },
    status: response.status,
  });
}
