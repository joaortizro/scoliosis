import { cookies } from "next/headers";
import { NextResponse } from "next/server";

const DEMO_SESSION_COOKIE = "spineview_demo_session";

export async function GET() {
  const cookieStore = await cookies();
  const isAuthenticated =
    cookieStore.get(DEMO_SESSION_COOKIE)?.value === "authenticated";

  return NextResponse.json({ isAuthenticated });
}
