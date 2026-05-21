"use client";

import { FormEvent, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";

export function LoginForm() {
  const searchParams = useSearchParams();
  const nextPath = searchParams.get("next") ?? "/prediction";
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [error, setError] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setStatus("loading");
    setError("");

    const response = await fetch("/api/demo-login", {
      body: JSON.stringify({ username, password }),
      headers: { "Content-Type": "application/json" },
      method: "POST",
    });

    if (!response.ok) {
      const payload = (await response.json().catch(() => null)) as {
        message?: string;
      } | null;
      setError(payload?.message ?? "Unable to sign in.");
      setStatus("error");
      return;
    }

    window.location.href = nextPath.startsWith("/") ? nextPath : "/prediction";
  }

  return (
    <Card className="mt-8 p-5 sm:p-6">
      <form className="grid gap-5" onSubmit={handleSubmit}>
        <label className="grid gap-2">
          <span className="text-sm font-semibold text-[#0d1620]">
            Username
          </span>
          <input
            autoComplete="username"
            className="h-11 rounded-full bg-[#f2f8ff] px-4 text-sm text-[#0d1620] outline-none ring-1 ring-[#c7c6b7]/45 transition focus:bg-white focus:ring-2 focus:ring-[#007ae5]/30"
            onChange={(event) => setUsername(event.target.value)}
            suppressHydrationWarning
            type="text"
            value={username}
          />
        </label>
        <label className="grid gap-2">
          <span className="text-sm font-semibold text-[#0d1620]">
            Password
          </span>
          <input
            autoComplete="current-password"
            className="h-11 rounded-full bg-[#f2f8ff] px-4 text-sm text-[#0d1620] outline-none ring-1 ring-[#c7c6b7]/45 transition focus:bg-white focus:ring-2 focus:ring-[#007ae5]/30"
            onChange={(event) => setPassword(event.target.value)}
            suppressHydrationWarning
            type="password"
            value={password}
          />
        </label>
        {status === "error" ? (
          <p className="rounded-2xl bg-[#fff0ed] px-4 py-3 text-sm leading-6 text-[#9a2600]">
            {error}
          </p>
        ) : null}
        <Button disabled={status === "loading"} type="submit">
          {status === "loading" ? "Signing in..." : "Sign in"}
        </Button>
      </form>
    </Card>
  );
}
