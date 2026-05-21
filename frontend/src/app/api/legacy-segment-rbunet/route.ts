import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const apiBaseUrl =
    process.env.LEGACY_BACKEND_API_BASE_URL?.replace(/\/$/, "") ??
    process.env.RBUNET_BACKEND_API_BASE_URL?.replace(/\/$/, "");

  if (!apiBaseUrl) {
    return NextResponse.json(
      { detail: "Missing LEGACY_BACKEND_API_BASE_URL environment variable." },
      { status: 500 },
    );
  }

  const formData = await request.formData();
  const response = await fetch(
    `${apiBaseUrl}/segment/rbunet?return_image=true`,
    {
      body: formData,
      method: "POST",
    },
  ).catch(() => null);

  if (!response) {
    return NextResponse.json(
      { detail: "Unable to reach the legacy segmentation API." },
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
