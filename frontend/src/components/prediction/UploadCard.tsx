"use client";

import {
  ChangeEvent,
  DragEvent,
  FormEvent,
  useEffect,
  useRef,
  useState,
} from "react";
import type { CSSProperties } from "react";
import { PREDICTION_MODEL_OPTIONS, predictScoliosisImage } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { PredictionResult } from "@/components/prediction/PredictionResult";
import { PROJECT_DISCLAIMER } from "@/lib/constants";
import type {
  PredictionModelKey,
  PredictionResultsByModel,
} from "@/types/prediction";

type SampleImage = {
  alt: string;
  label: string;
  src: string;
};

type UploadCardProps = {
  sampleImages?: SampleImage[];
};

type RequestState =
  | { status: "idle"; error: null }
  | { status: "loading"; error: null }
  | { status: "error"; error: string };

const INITIAL_VISIBLE_LAYERS = {
  binary: true,
  multiclass: true,
};

function responseHasLayer(
  response: PredictionResultsByModel[PredictionModelKey],
  layer: "binary" | "multiclass",
) {
  if (!response?.results) {
    return false;
  }

  if (response.results[layer]?.segments?.length) {
    return true;
  }

  return Object.values(response.results).some(
    (result) => result.type === layer && Boolean(result.segments?.length),
  );
}

function getFilenameFromSrc(src: string) {
  return decodeURIComponent(src.split("/").pop() ?? "sample-xray.jpg");
}

export function UploadCard({ sampleImages = [] }: UploadCardProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] =
    useState<PredictionModelKey>("full");
  const [predictions, setPredictions] = useState<PredictionResultsByModel>({});
  const [visibleLayers, setVisibleLayers] = useState(INITIAL_VISIBLE_LAYERS);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [isPanelOpen, setIsPanelOpen] = useState(true);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [requestState, setRequestState] = useState<RequestState>({
    status: "idle",
    error: null,
  });

  const hasFullPrediction = Boolean(predictions.full);
  const hasBinaryPrediction = Boolean(predictions.binary);
  const hasMulticlassPrediction = Boolean(predictions.multiclass);
  const hasAllPartialPredictions = hasBinaryPrediction && hasMulticlassPrediction;
  const availableLayers = (["multiclass", "binary"] as const).filter((layer) =>
    predictions.full
      ? responseHasLayer(predictions.full, layer)
      : responseHasLayer(predictions[layer], layer),
  );

  function isModelDisabled(model: PredictionModelKey) {
    if (!selectedFile || requestState.status === "loading" || hasFullPrediction) {
      return true;
    }

    if (model === "full") {
      return hasBinaryPrediction || hasMulticlassPrediction;
    }

    if (model === "binary") {
      return hasBinaryPrediction || hasAllPartialPredictions;
    }

    return hasMulticlassPrediction || hasAllPartialPredictions;
  }

  const submitDisabled =
    !selectedFile ||
    requestState.status === "loading" ||
    isModelDisabled(selectedModel);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  function resetWorkspace() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    setSelectedFile(null);
    setPreviewUrl(null);
    setSelectedModel("full");
    setPredictions({});
    setVisibleLayers(INITIAL_VISIBLE_LAYERS);
    setShowBoundingBoxes(true);
    setPan({ x: 0, y: 0 });
    setZoom(1);
    setRequestState({ status: "idle", error: null });
  }

  function loadFile(file: File | null) {
    if (!file) {
      resetWorkspace();
      return;
    }

    if (!["image/jpeg", "image/png"].includes(file.type)) {
      resetWorkspace();
      setRequestState({
        status: "error",
        error: "Please upload a JPG or PNG image.",
      });
      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setSelectedModel("full");
    setPredictions({});
    setVisibleLayers(INITIAL_VISIBLE_LAYERS);
    setShowBoundingBoxes(true);
    setPan({ x: 0, y: 0 });
    setZoom(1);
    setRequestState({ status: "idle", error: null });
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    loadFile(file);
  }

  async function loadSampleImage(sampleImage: SampleImage) {
    setRequestState({ status: "loading", error: null });

    try {
      const response = await fetch(sampleImage.src);

      if (!response.ok) {
        throw new Error("Unable to load this sample image.");
      }

      const blob = await response.blob();
      const file = new File([blob], getFilenameFromSrc(sampleImage.src), {
        type: blob.type || "image/jpeg",
      });

      loadFile(file);
    } catch (error) {
      setRequestState({
        status: "error",
        error:
          error instanceof Error
            ? error.message
            : "Unable to load this sample image.",
      });
    }
  }

  function handleDragOver(event: DragEvent<HTMLDivElement>) {
    if (!Array.from(event.dataTransfer.types).includes("Files")) {
      return;
    }

    event.preventDefault();
    setIsDragging(true);
  }

  function handleDragLeave(event: DragEvent<HTMLDivElement>) {
    if (event.currentTarget.contains(event.relatedTarget as Node | null)) {
      return;
    }

    setIsDragging(false);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();

    if (!event.dataTransfer.files.length) {
      setIsDragging(false);
      return;
    }

    setIsDragging(false);
    loadFile(event.dataTransfer.files[0] ?? null);
  }

  function adjustZoom(amount: number) {
    setZoom((currentZoom) =>
      Math.min(2.5, Math.max(0.5, Number((currentZoom + amount).toFixed(2)))),
    );
  }

  function resetView() {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!selectedFile) {
      setRequestState({
        status: "error",
        error: "Please select an image before submitting.",
      });
      return;
    }

    if (isModelDisabled(selectedModel)) {
      setRequestState({
        status: "error",
        error: "Upload a new image or choose an available model.",
      });
      return;
    }

    setRequestState({ status: "loading", error: null });

    try {
      const data = await predictScoliosisImage(selectedFile, selectedModel);
      setPredictions((currentPredictions) => ({
        ...currentPredictions,
        [selectedModel]: data,
      }));
      setVisibleLayers((currentLayers) => ({
        ...currentLayers,
        binary: selectedModel === "binary" || selectedModel === "full"
          ? true
          : currentLayers.binary,
        multiclass:
          selectedModel === "multiclass" || selectedModel === "full"
            ? true
            : currentLayers.multiclass,
      }));
      setSelectedModel((currentModel) => {
        if (currentModel === "multiclass") {
          return "binary";
        }

        if (currentModel === "binary") {
          return "multiclass";
        }

        return currentModel;
      });
      setRequestState({ status: "idle", error: null });
    } catch (error) {
      setRequestState({
        status: "error",
        error:
          error instanceof Error
            ? error.message
            : "Unexpected prediction error.",
      });
    }
  }

  return (
    <form
      className="grid h-full min-h-0 grid-cols-1 transition-[grid-template-columns] duration-200 lg:grid-cols-[minmax(0,1fr)_var(--panel-width)]"
      onSubmit={handleSubmit}
      style={
        {
          "--panel-width": isPanelOpen ? "360px" : "0px",
        } as CSSProperties
      }
    >
      <div className="relative flex min-h-0 bg-white p-3">
        <div
          className={`relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-3xl p-5 transition ${
            isDragging
              ? "bg-[#f2f8ff] ring-2 ring-[#007ae5]/45"
              : "bg-[#f2f8ff] ring-1 ring-[#c7c6b7]/40"
          }`}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <PredictionResult
            imageUrl={previewUrl}
            onPanChange={setPan}
            pan={pan}
            predictions={predictions}
            showBoundingBoxes={showBoundingBoxes}
            visibleLayers={visibleLayers}
            zoom={zoom}
          />

          {!isPanelOpen ? (
            <button
              aria-label="Open control panel"
              className="absolute right-4 top-4 grid h-10 w-10 place-items-center rounded-full bg-white/95 text-lg font-semibold text-[#1c3f9a] shadow-sm shadow-[#073f73]/10 transition hover:bg-[#f2f8ff]"
              onClick={() => setIsPanelOpen(true)}
              type="button"
            >
              ‹
            </button>
          ) : null}

          <div className="absolute left-4 top-1/2 grid -translate-y-1/2 gap-2 rounded-full bg-white/95 p-1 shadow-sm shadow-[#073f73]/10 ring-1 ring-[#c7c6b7]/35">
            <button
              aria-label="Zoom in"
              className="grid h-9 w-9 place-items-center rounded-full text-lg font-semibold text-[#0d1620] transition hover:bg-[#f2f8ff]"
              onClick={() => adjustZoom(0.2)}
              type="button"
            >
              +
            </button>
            <button
              aria-label="Zoom out"
              className="grid h-9 w-9 place-items-center rounded-full text-lg font-semibold text-[#0d1620] transition hover:bg-[#f2f8ff]"
              onClick={() => adjustZoom(-0.2)}
              type="button"
            >
              -
            </button>
            <button
              aria-label="Reset zoom"
              className="grid h-9 w-9 place-items-center rounded-full text-lg font-bold text-[#182433] transition hover:bg-[#f2f8ff]"
              onClick={resetView}
              type="button"
            >
              ↺
            </button>
          </div>
        </div>

        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex justify-center px-4 pb-4">
          <div className="pointer-events-auto flex w-full max-w-md items-center justify-center gap-2 rounded-full bg-white/95 p-2 shadow-lg shadow-[#073f73]/10 ring-1 ring-[#c7c6b7]/35">
            <input
              ref={fileInputRef}
              className="sr-only"
              type="file"
              accept="image/png,image/jpeg"
              onChange={handleFileChange}
            />
            <button
              className="inline-flex h-10 items-center justify-center rounded-full bg-[#f2f8ff] px-4 text-sm font-semibold text-[#1c3f9a] transition hover:bg-[#dcedff]"
              onClick={() => fileInputRef.current?.click()}
              type="button"
            >
              Choose file
            </button>
            <Button className="h-10 flex-1" disabled={submitDisabled} type="submit">
              {requestState.status === "loading"
                ? "Segmenting..."
                : `Run ${PREDICTION_MODEL_OPTIONS.find((option) => option.key === selectedModel)?.label ?? "model"}`}
            </Button>
            {selectedFile ? (
              <button
                className="inline-flex h-10 items-center justify-center rounded-full bg-[#fff4ed] px-4 text-sm font-semibold text-[#9a3500] transition hover:bg-[#ffe6d8]"
                onClick={resetWorkspace}
                type="button"
              >
                Restart
              </button>
            ) : null}
          </div>
        </div>
      </div>

      <aside
        className={`min-h-0 overflow-hidden bg-white transition-opacity duration-200 ${
          isPanelOpen ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
      >
        <div className="flex h-full min-w-0 flex-col">
          <div className="min-w-0 border-b border-[#c7c6b7]/40 p-5">
            <div className="flex min-w-0 items-start justify-between gap-3">
              <div className="min-w-0">
                <h1 className="text-2xl font-semibold leading-tight text-[#1c3f9a]">
                  Upload X-ray
                </h1>
                <p className="mt-2 text-sm leading-6 text-[#182433]/70">
                  Select a JPG or PNG frontal spine radiograph to inspect model
                  segmentation layers.
                </p>
              </div>
              <button
                aria-label="Collapse control panel"
                className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-[#f2f8ff] text-lg font-semibold text-[#1c3f9a] transition hover:bg-[#dcedff]"
                onClick={() => setIsPanelOpen(false)}
                type="button"
              >
                ›
              </button>
            </div>
          </div>

          <div className="grid min-h-0 min-w-0 flex-1 content-start gap-5 overflow-x-hidden overflow-y-auto p-5">
            {sampleImages.length ? (
              <fieldset className="grid min-w-0 gap-2">
                <legend className="text-sm font-semibold text-[#0d1620]">
                  Examples
                </legend>
                <div className="grid min-w-0 grid-cols-3 gap-2">
                  {sampleImages.map((sampleImage) => (
                    <button
                      className="group min-w-0 overflow-hidden rounded-2xl bg-[#f2f8ff] text-left ring-1 ring-[#c7c6b7]/35 transition hover:bg-[#dcedff] hover:ring-[#007ae5]/35 focus:outline-none focus:ring-2 focus:ring-[#007ae5]/35"
                      key={sampleImage.src}
                      onClick={() => void loadSampleImage(sampleImage)}
                      type="button"
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        alt={sampleImage.alt}
                        className="h-24 w-full object-cover grayscale transition group-hover:grayscale-0"
                        src={sampleImage.src}
                      />
                      <span className="block truncate px-2 py-2 text-xs font-semibold text-[#1c3f9a]">
                        {sampleImage.label}
                      </span>
                    </button>
                  ))}
                </div>
              </fieldset>
            ) : null}

            <fieldset className="grid min-w-0 gap-2">
              <legend className="text-sm font-semibold text-[#0d1620]">
                Models
              </legend>
              <div className="grid min-w-0 gap-2">
                {PREDICTION_MODEL_OPTIONS.map((option) => (
                  <label
                    className={`min-w-0 rounded-2xl p-3 transition ${
                      isModelDisabled(option.key)
                        ? "cursor-not-allowed bg-[#fbfaf0] opacity-55"
                        : selectedModel === option.key
                          ? "bg-[#f2f8ff] ring-2 ring-[#007ae5]/35"
                          : "cursor-pointer bg-white ring-1 ring-[#c7c6b7]/35 hover:bg-[#f2f8ff]"
                    }`}
                    key={option.key}
                  >
                    <input
                      checked={selectedModel === option.key}
                      className="sr-only"
                      disabled={isModelDisabled(option.key)}
                      name="prediction-model"
                      onChange={() => setSelectedModel(option.key)}
                      suppressHydrationWarning
                      type="radio"
                      value={option.key}
                    />
                    <span className="block min-w-0 break-words text-sm font-semibold text-[#0d1620]">
                      {option.label}
                    </span>
                    <span className="mt-1 block min-w-0 break-words text-xs leading-5 text-[#182433]/70">
                      {option.description}
                    </span>
                  </label>
                ))}
              </div>
            </fieldset>

            {availableLayers.length ? (
              <fieldset className="grid min-w-0 gap-2">
                <legend className="text-sm font-semibold text-[#0d1620]">
                  Layers
                </legend>
                <div className="grid min-w-0 gap-2">
                  {availableLayers.map((layer) => (
                    <label
                      className="flex min-h-9 min-w-0 items-center gap-2 rounded-full bg-[#f2f8ff] px-3 py-2 text-sm font-semibold text-[#1c3f9a]"
                      key={layer}
                    >
                      <input
                        checked={visibleLayers[layer]}
                        className="h-4 w-4 accent-[#007ae5]"
                        onChange={(event) =>
                          setVisibleLayers((currentLayers) => ({
                            ...currentLayers,
                            [layer]: event.target.checked,
                          }))
                        }
                        type="checkbox"
                      />
                      <span className="min-w-0 break-words">
                        {layer === "multiclass"
                          ? "Vertebra labels"
                          : "Binary mask"}
                      </span>
                    </label>
                  ))}
                  {availableLayers.includes("multiclass") ? (
                    <label className="flex min-h-9 min-w-0 items-center gap-2 rounded-full bg-[#f2f8ff] px-3 py-2 text-sm font-semibold text-[#1c3f9a]">
                      <input
                        checked={showBoundingBoxes}
                        className="h-4 w-4 accent-[#007ae5]"
                        onChange={(event) =>
                          setShowBoundingBoxes(event.target.checked)
                        }
                        type="checkbox"
                      />
                      <span className="min-w-0 break-words">Bounding boxes</span>
                    </label>
                  ) : null}
                </div>
              </fieldset>
            ) : null}

            <div className="min-w-0 rounded-2xl bg-[#fbfaf0] p-3 ring-1 ring-[#c7c6b7]/35">
              <p className="text-xs font-semibold uppercase tracking-wide text-[#182433]/55">
                Current file
              </p>
              <p className="mt-1 break-words text-sm font-semibold text-[#0d1620]">
                {selectedFile?.name ?? "No image selected"}
              </p>
            </div>

            {requestState.status === "error" ? (
              <p className="rounded-2xl bg-[#fff0ed] px-4 py-3 text-sm leading-6 text-[#9a2600]">
                {requestState.error}
              </p>
            ) : null}
          </div>

          <div className="min-w-0 border-t border-[#c7c6b7]/40 bg-[#f2f8ff] px-5 py-4 text-xs leading-5 text-[#182433]/70">
            <p className="font-medium text-[#0d1620]">
              Research frontend for AI-powered scoliosis detection.
            </p>
            <p className="mt-1">{PROJECT_DISCLAIMER}</p>
          </div>
        </div>
      </aside>
    </form>
  );
}
