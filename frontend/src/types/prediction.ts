export type PredictionResponse = {
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
