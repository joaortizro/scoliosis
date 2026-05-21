"use client";

import { useState } from "react";
import { Card } from "@/components/ui/Card";
import type {
  PredictionResponse,
  PredictionState,
  VertebraPrediction,
} from "@/types/prediction";

type PredictionResultProps = {
  originalImageUrl: string | null;
  state: PredictionState;
};

function getVertebrae(result: PredictionResponse): VertebraPrediction[] {
  return result.data?.vertebrae ?? result.vertebrae ?? [];
}

function getSegmentedImageUrl(result: PredictionResponse) {
  const imageBase64 = result.data?.image_base64 ?? result.image_base64;

  if (!imageBase64) {
    return null;
  }

  if (imageBase64.startsWith("data:image/")) {
    return imageBase64;
  }

  return `data:image/png;base64,${imageBase64}`;
}

function formatConfidence(value: unknown) {
  if (typeof value !== "number") {
    return "N/A";
  }

  return `${Math.round(value * 100)}%`;
}

function getResponseForDisplay(result: PredictionResponse) {
  return JSON.stringify(
    result,
    (key, value) =>
      key === "image_base64" && typeof value === "string"
        ? `${value.slice(0, 80)}...`
        : value,
    2,
  );
}

export function PredictionResult({
  originalImageUrl,
  state,
}: PredictionResultProps) {
  const [viewMode, setViewMode] = useState<"segmented" | "original">(
    "segmented",
  );

  if (state.status === "idle") {
    return (
      <Card className="p-5">
        <h3 className="text-lg font-semibold text-slate-950">Result</h3>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          Upload a JPG or PNG image and submit it to view the segmented result.
        </p>
      </Card>
    );
  }

  if (state.status === "loading") {
    return (
      <Card className="p-5">
        <h3 className="text-lg font-semibold text-slate-950">
          Segmenting image
        </h3>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          Sending the selected file to the vertebra segmentation API.
        </p>
      </Card>
    );
  }

  if (state.status === "error") {
    return (
      <Card className="border-red-200 bg-red-50 p-5">
        <h3 className="text-lg font-semibold text-red-950">
          Prediction failed
        </h3>
        <p className="mt-2 text-sm leading-6 text-red-800">{state.error}</p>
      </Card>
    );
  }

  const vertebrae = getVertebrae(state.data);
  const segmentedImageUrl = getSegmentedImageUrl(state.data);
  const activeImageUrl =
    viewMode === "segmented" ? segmentedImageUrl : originalImageUrl;

  return (
    <Card className="p-5 sm:p-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h3 className="text-lg font-semibold text-slate-950">
            Segmentation result
          </h3>
          <p className="mt-1 text-sm text-slate-600">
            {vertebrae.length
              ? `${vertebrae.length} vertebrae returned by the model.`
              : "Response received from the model."}
          </p>
        </div>
        <div className="grid grid-cols-2 rounded-md border border-slate-200 bg-slate-50 p-1 text-sm font-semibold">
          <button
            className={`rounded px-3 py-2 transition ${
              viewMode === "segmented"
                ? "bg-white text-[#0a5f9e] shadow-sm"
                : "text-slate-600"
            }`}
            disabled={!segmentedImageUrl}
            onClick={() => setViewMode("segmented")}
            type="button"
          >
            Result
          </button>
          <button
            className={`rounded px-3 py-2 transition ${
              viewMode === "original"
                ? "bg-white text-[#0a5f9e] shadow-sm"
                : "text-slate-600"
            }`}
            disabled={!originalImageUrl}
            onClick={() => setViewMode("original")}
            type="button"
          >
            Original
          </button>
        </div>
      </div>

      <div className="mt-5 flex min-h-[340px] items-center justify-center rounded-lg bg-slate-950 p-4">
        {activeImageUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            alt={
              viewMode === "segmented"
                ? "Segmented spine result"
                : "Original uploaded spine X-ray"
            }
            className="max-h-[560px] w-auto rounded-md object-contain"
            src={activeImageUrl}
          />
        ) : (
          <p className="text-center text-sm leading-6 text-slate-300">
            The API response did not include a segmented image.
          </p>
        )}
      </div>

      {vertebrae.length ? (
        <div className="mt-5 overflow-hidden rounded-lg border border-slate-200">
          <div className="grid grid-cols-[1fr_1fr_1fr] bg-slate-50 px-4 py-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            <span>Label</span>
            <span>Confidence</span>
            <span>Area</span>
          </div>
          <div className="max-h-72 overflow-auto">
            {vertebrae.map((vertebra, index) => (
              <div
                className="grid grid-cols-[1fr_1fr_1fr] border-t border-slate-100 px-4 py-3 text-sm text-slate-700"
                key={`${vertebra.label ?? "vertebra"}-${index}`}
              >
                <span className="font-semibold text-slate-950">
                  {vertebra.label ?? "Unknown"}
                </span>
                <span>{formatConfidence(vertebra.confidence)}</span>
                <span>
                  {typeof vertebra.area_px === "number"
                    ? `${vertebra.area_px} px`
                    : "N/A"}
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <details className="mt-4">
        <summary className="cursor-pointer text-sm font-medium text-cyan-800">
          Raw API response
        </summary>
        <pre className="mt-3 max-h-72 overflow-auto rounded-md bg-slate-950 p-4 text-xs leading-5 text-slate-100">
          {getResponseForDisplay(state.data)}
        </pre>
      </details>
    </Card>
  );
}
