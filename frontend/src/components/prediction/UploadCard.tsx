"use client";

import { ChangeEvent, FormEvent, useEffect, useState } from "react";
import { predictScoliosisImage } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { PredictionResult } from "@/components/prediction/PredictionResult";
import type { PredictionState } from "@/types/prediction";

export function UploadCard() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [predictionState, setPredictionState] = useState<PredictionState>({
    status: "idle",
    data: null,
    error: null,
  });

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;

    if (file && !["image/jpeg", "image/png"].includes(file.type)) {
      setSelectedFile(null);
      setPreviewUrl(null);
      setPredictionState({
        status: "error",
        data: null,
        error: "Please upload a JPG or PNG image.",
      });
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(file ? URL.createObjectURL(file) : null);
    setPredictionState({ status: "idle", data: null, error: null });
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!selectedFile) {
      setPredictionState({
        status: "error",
        data: null,
        error: "Please select an image before submitting.",
      });
      return;
    }

    setPredictionState({ status: "loading", data: null, error: null });

    try {
      const data = await predictScoliosisImage(selectedFile);
      setPredictionState({ status: "success", data, error: null });
    } catch (error) {
      setPredictionState({
        status: "error",
        data: null,
        error:
          error instanceof Error
            ? error.message
            : "Unexpected prediction error.",
      });
    }
  }

  return (
    <div className="mx-auto grid w-full max-w-4xl gap-4">
      <Card className="p-5 sm:p-6">
        <form className="grid gap-5" onSubmit={handleSubmit}>
          <label className="grid gap-2">
            <span className="text-sm font-semibold text-slate-800">
              X-ray image
            </span>
            <input
              className="block w-full rounded-md border border-slate-300 bg-white text-sm text-slate-700 file:mr-4 file:border-0 file:bg-slate-100 file:px-4 file:py-3 file:text-sm file:font-semibold file:text-slate-800 hover:file:bg-slate-200"
              type="file"
              accept="image/png,image/jpeg"
              onChange={handleFileChange}
            />
          </label>

          <div className="flex min-h-[320px] items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50 p-4">
            {previewUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                alt="Selected X-ray preview"
                className="max-h-[420px] w-auto rounded-md object-contain"
                src={previewUrl}
              />
            ) : (
              <p className="max-w-sm text-center text-sm leading-6 text-slate-500">
                Selected image preview will appear here before submission.
              </p>
            )}
          </div>

          <Button disabled={predictionState.status === "loading"} type="submit">
            {predictionState.status === "loading"
              ? "Segmenting..."
              : "Submit image"}
          </Button>
        </form>
      </Card>
      <PredictionResult originalImageUrl={previewUrl} state={predictionState} />
    </div>
  );
}
