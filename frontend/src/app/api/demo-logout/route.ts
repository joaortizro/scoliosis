import { NextResponse } from "next/server";

const DEMO_SESSION_COOKIE = "spineview_demo_session";

export async function POST() {
  const response = NextResponse.json({ ok: true });
  response.cookies.set(DEMO_SESSION_COOKIE, "", {
    expires: new Date(0),
    path: "/",
  });

  return response;
}
