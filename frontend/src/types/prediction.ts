export type PredictionModelKey = "binary" | "multiclass" | "full";

export type PredictionSegment = {
  id?: number;
  class_id?: number;
  label?: string;
  bbox?: [number, number, number, number] | number[];
  area?: number;
  area_px?: number;
  polygon?: [number, number][] | number[][];
  confidence?: number;
  [key: string]: unknown;
};

export type ModelPredictionResult = {
  type?: string;
  classes_detected?: string[];
  segments?: PredictionSegment[];
  [key: string]: unknown;
};

export type VertebraPrediction = {
  label?: string;
  confidence?: number;
  centroid_x?: number;
  centroid_y?: number;
  area_px?: number;
  source?: string;
  [key: string]: unknown;
};

export type PredictionResponse = {
  image_width?: number;
  image_height?: number;
  results?: Record<string, ModelPredictionResult>;
  data?: {
    vertebrae?: VertebraPrediction[];
    image_base64?: string;
    [key: string]: unknown;
  };
  vertebrae?: VertebraPrediction[];
  image_base64?: string;
  prediction?: string;
  label?: string;
  confidence?: number;
  probability?: number;
  cobb_angle?: number;
  message?: string;
  [key: string]: unknown;
};

export type PredictionState =
  | { status: "idle"; data: null; error: null }
  | { status: "loading"; data: null; error: null }
  | { status: "success"; data: PredictionResponse; error: null }
  | { status: "error"; data: null; error: string };

export type PredictionResultsByModel = Partial<
  Record<PredictionModelKey, PredictionResponse>
>;
