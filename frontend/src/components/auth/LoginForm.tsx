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
          <span className="text-sm font-semibold text-slate-800">
            Username
          </span>
          <input
            autoComplete="username"
            className="h-11 rounded-md border border-slate-300 bg-white px-3 text-sm text-slate-900 outline-none transition focus:border-[#0a5f9e] focus:ring-2 focus:ring-[#0a5f9e]/20"
            onChange={(event) => setUsername(event.target.value)}
            suppressHydrationWarning
            type="text"
            value={username}
          />
        </label>
        <label className="grid gap-2">
          <span className="text-sm font-semibold text-slate-800">
            Password
          </span>
          <input
            autoComplete="current-password"
            className="h-11 rounded-md border border-slate-300 bg-white px-3 text-sm text-slate-900 outline-none transition focus:border-[#0a5f9e] focus:ring-2 focus:ring-[#0a5f9e]/20"
            onChange={(event) => setPassword(event.target.value)}
            suppressHydrationWarning
            type="password"
            value={password}
          />
        </label>
        {status === "error" ? (
          <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm leading-6 text-red-800">
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
