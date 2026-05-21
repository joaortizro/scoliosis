import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { LoginForm } from "@/components/auth/LoginForm";
import { Badge } from "@/components/ui/Badge";
import { PROJECT_DISCLAIMER } from "@/lib/constants";

const DEMO_SESSION_COOKIE = "spineview_demo_session";

export default async function LoginPage() {
  const cookieStore = await cookies();
  const isAuthenticated =
    cookieStore.get(DEMO_SESSION_COOKIE)?.value === "authenticated";

  if (isAuthenticated) {
    redirect("/prediction");
  }

  return (
    <section className="bg-white">
      <div className="mx-auto flex min-h-[calc(100vh-144px)] w-full max-w-5xl flex-col items-center justify-center px-5 py-10 lg:px-8">
        <div className="w-full max-w-md">
          <div className="text-center">
            <Badge>Demo access</Badge>
            <h1 className="mt-5 text-3xl font-semibold leading-tight text-[#1c3f9a] sm:text-4xl">
              Sign in to use Flash Prediction.
            </h1>
            <p className="mt-4 text-base leading-7 text-[#182433]/75">
              This demo login protects the research workspace while keeping the
              public project page open.
            </p>
          </div>
          <LoginForm />
          <p className="mt-5 rounded-2xl bg-[#fff4ed] p-4 text-sm leading-6 text-[#9a3500]">
            {PROJECT_DISCLAIMER}
          </p>
        </div>
      </div>
    </section>
  );
}
