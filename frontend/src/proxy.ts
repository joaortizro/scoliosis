import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const DEMO_SESSION_COOKIE = "spineview_demo_session";

export function proxy(request: NextRequest) {
  const isAuthenticated =
    request.cookies.get(DEMO_SESSION_COOKIE)?.value === "authenticated";

  if (!isAuthenticated) {
    const loginUrl = new URL("/login", request.url);
    loginUrl.searchParams.set("next", request.nextUrl.pathname);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/prediction/:path*"],
};
