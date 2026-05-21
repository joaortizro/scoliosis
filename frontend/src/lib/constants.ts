export const PROJECT_DISCLAIMER =
  "This tool is for research and educational purposes only. It is not a medical diagnosis tool.";

export const DEFAULT_API_BASE_URL = "";

export const BRAND_COLORS = {
  orange: "#ff5c00",
  orangeDark: "#cc4900",
  orangeLight: "#ff8a3d",
  primaryBlue: "#007ae5",
  primaryBlueDark: "#005fb3",
  primaryBlueLight: "#4ca6f0",
  darkBlue: "#1c3f9a",
  alternateDarkBlue: "#073f73",
  white: "#f5f4df",
  black: "#0d1620",
  blackSoft: "#182433",
  gray: "#c7c6b7",
  grayLight: "#e3e1d0",
  medicalTeal: "#00d1c1",
} as const;

export const GENERAL_SPINE_COLORS = {
  border: "#1C3F98",
  fill: "#0078E5",
} as const;

export const VERTEBRA_COLORS = {
  T1: { fill: "#8F1634", border: "#FF4D7D" },
  T2: { fill: "#A83222", border: "#FF735C" },
  T3: { fill: "#A84F08", border: "#FF9E2C" },
  T4: { fill: "#9A7100", border: "#FFD447" },
  T5: { fill: "#7A8500", border: "#E6FF45" },
  T6: { fill: "#4B7F13", border: "#B6FF4D" },
  T7: { fill: "#1D7A2E", border: "#6DFF7F" },
  T8: { fill: "#0C7655", border: "#55FFB8" },
  T9: { fill: "#00706F", border: "#4DFFF0" },
  T10: { fill: "#00698F", border: "#52DFFF" },
  T11: { fill: "#175CA5", border: "#64B8FF" },
  T12: { fill: "#304AAE", border: "#7F96FF" },
  L1: { fill: "#5039A8", border: "#A995FF" },
  L2: { fill: "#6B2A9A", border: "#C97BFF" },
  L3: { fill: "#842184", border: "#F06CFF" },
  L4: { fill: "#981D62", border: "#FF66C7" },
  L5: { fill: "#9C1E48", border: "#FF5F96" },
} as const;

export const EXPORT_VERTEBRA_COLORS = {
  T1: "#F2D10C",
  T2: "#EBF20C",
  T3: "#C4F20C",
  T4: "#9CF20C",
  T5: "#75F20C",
  T6: "#4DF20C",
  T7: "#26F20C",
  T8: "#0CF219",
  T9: "#0CF240",
  T10: "#0CF268",
  T11: "#0CF28F",
  T12: "#0CF2B7",
  L1: "#0CF2DE",
  L2: "#0CDEF2",
  L3: "#0CB7F2",
  L4: "#0C8FF2",
  L5: "#0C68F2",
} as const;

export type VertebraLabel = keyof typeof VERTEBRA_COLORS;

export function getVertebraColors(label: string | undefined) {
  const normalizedLabel = label?.trim().toUpperCase() as
    | VertebraLabel
    | undefined;

  if (normalizedLabel && normalizedLabel in VERTEBRA_COLORS) {
    return VERTEBRA_COLORS[normalizedLabel];
  }

  return GENERAL_SPINE_COLORS;
}

export function getExportVertebraColor(label: string | undefined) {
  const normalizedLabel = label?.trim().toUpperCase() as
    | VertebraLabel
    | undefined;

  if (normalizedLabel && normalizedLabel in EXPORT_VERTEBRA_COLORS) {
    return EXPORT_VERTEBRA_COLORS[normalizedLabel];
  }

  return GENERAL_SPINE_COLORS.fill;
}
