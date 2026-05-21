"use client";

import { useState } from "react";
import type { PointerEvent } from "react";
import {
  formatConfidenceDecimal,
  getOverlayLayers,
  getImageSize,
  getPolygonPoints,
  getSegmentColors,
  getSegmentKey,
  getSegmentLabel,
} from "@/lib/prediction-results";
import type { PredictionResultsByModel } from "@/types/prediction";

type PredictionResultProps = {
  imageUrl: string | null;
  hiddenSegments?: Record<string, boolean>;
  isLoading?: boolean;
  predictions?: PredictionResultsByModel;
  showBoundingBoxes: boolean;
  showBoxConfidence: boolean;
  showBoxLabels: boolean;
  visibleLayers: Record<"binary" | "multiclass", boolean>;
  zoom: number;
  onPanChange: (pan: { x: number; y: number }) => void;
  pan: { x: number; y: number };
};

export function PredictionResult({
  hiddenSegments = {},
  imageUrl,
  isLoading = false,
  onPanChange,
  pan,
  predictions = {},
  showBoxConfidence,
  showBoxLabels,
  showBoundingBoxes,
  visibleLayers,
  zoom,
}: PredictionResultProps) {
  const [dragStart, setDragStart] = useState<{
    panX: number;
    panY: number;
    pointerId: number;
    x: number;
    y: number;
  } | null>(null);
  const imageSize = getImageSize(predictions);
  const overlayLayers = getOverlayLayers(predictions, visibleLayers);

  function handlePointerDown(event: PointerEvent<HTMLDivElement>) {
    if (zoom <= 1) {
      return;
    }

    event.currentTarget.setPointerCapture(event.pointerId);
    setDragStart({
      panX: pan.x,
      panY: pan.y,
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
    });
  }

  function handlePointerMove(event: PointerEvent<HTMLDivElement>) {
    if (!dragStart) {
      return;
    }

    onPanChange({
      x: dragStart.panX + event.clientX - dragStart.x,
      y: dragStart.panY + event.clientY - dragStart.y,
    });
  }

  function handlePointerUp(event: PointerEvent<HTMLDivElement>) {
    if (dragStart?.pointerId === event.pointerId) {
      setDragStart(null);
    }
  }

  if (!imageUrl) {
    return (
      <div className="grid place-items-center text-center">
        <div className="grid h-16 w-16 place-items-center rounded-full border-2 border-dashed border-[#007ae5]/30 bg-white text-3xl text-[#007ae5] shadow-sm shadow-[#073f73]/10">
          +
        </div>
        <p className="mt-4 max-w-xs text-sm font-semibold text-[#0d1620]">
          Drop a frontal spine radiograph here
        </p>
        <p className="mt-1 max-w-xs text-sm leading-6 text-[#182433]/65">
          JPG and PNG images are supported.
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full w-full items-center justify-center overflow-auto p-4">
      <div
        className={`relative inline-block max-w-full origin-center select-none ${
          zoom > 1 ? "cursor-grab active:cursor-grabbing" : ""
        }`}
        onPointerCancel={handlePointerUp}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        style={{
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          transition: dragStart ? "none" : "transform 160ms ease",
        }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          alt="Selected X-ray preview"
          className="max-h-[calc(100vh-12rem)] w-auto max-w-full rounded-2xl object-contain shadow-sm shadow-[#073f73]/10"
          draggable={false}
          src={imageUrl}
        />
        {imageSize && overlayLayers.length ? (
          <svg
            aria-label="Prediction geometry overlay"
            className="pointer-events-none absolute inset-0 h-full w-full"
            preserveAspectRatio="none"
            viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
          >
            {overlayLayers.map((layer) =>
              layer.segments.map((segment, index) => {
                const segmentKey = getSegmentKey(layer.key, segment, index);

                if (hiddenSegments[segmentKey]) {
                  return null;
                }

                const colors = getSegmentColors(segment, layer.key);
                const polygonPoints = getPolygonPoints(segment)
                  .map((point) => point.join(","))
                  .join(" ");
                const bbox = segment.bbox;
                const showBox = layer.key === "multiclass" && showBoundingBoxes;
                const confidence = formatConfidenceDecimal(segment);
                const boxLabel = [
                  showBoxLabels ? getSegmentLabel(segment) : null,
                  showBoxConfidence ? confidence : null,
                ]
                  .filter(Boolean)
                  .join(" ");

                return (
                  <g key={segmentKey}>
                    {polygonPoints ? (
                      <polygon
                        fill={colors.fill}
                        fillOpacity={layer.key === "binary" ? "0.28" : "0.3"}
                        points={polygonPoints}
                        stroke={colors.border}
                        strokeLinejoin="round"
                        strokeWidth={layer.key === "binary" ? "2" : "2.5"}
                      />
                    ) : null}
                    {showBox && bbox && bbox.length === 4 ? (
                      <rect
                        fill="none"
                        height={Math.max(0, bbox[3] - bbox[1])}
                        stroke={colors.border}
                        strokeWidth="2.5"
                        width={Math.max(0, bbox[2] - bbox[0])}
                        x={bbox[0]}
                        y={bbox[1]}
                      />
                    ) : null}
                    {showBox && boxLabel && bbox && bbox.length === 4 ? (
                      <text
                        fill="white"
                        fontSize="22"
                        fontWeight="700"
                        paintOrder="stroke"
                        stroke="#0d1620"
                        strokeWidth="5"
                        x={bbox[0]}
                        y={Math.max(24, bbox[1] - 8)}
                      >
                        {boxLabel}
                      </text>
                    ) : null}
                  </g>
                );
              }),
            )}
          </svg>
        ) : null}
        {isLoading ? (
          <div className="absolute inset-0 grid place-items-center rounded-2xl bg-[#0d1620]/35 px-6 text-center text-white backdrop-blur-[1px]">
            <div className="rounded-2xl bg-[#0d1620]/80 px-5 py-4 shadow-lg shadow-[#073f73]/20">
              <div className="mx-auto mb-3 flex w-16 justify-between">
                <span className="h-2 w-2 animate-pulse rounded-full bg-white [animation-delay:0ms]" />
                <span className="h-2 w-2 animate-pulse rounded-full bg-white [animation-delay:160ms]" />
                <span className="h-2 w-2 animate-pulse rounded-full bg-white [animation-delay:320ms]" />
              </div>
              <p className="text-sm font-semibold">Analyzing image</p>
              <p className="mt-1 text-xs leading-5 text-white/80">
                This may take a few seconds. Please wait.
              </p>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
