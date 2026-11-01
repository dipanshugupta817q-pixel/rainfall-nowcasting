"use client";

import { useEffect, useRef, useState } from "react";
import {
  Map,
  setWorkerUrl,
  NavigationControl,
  ScaleControl,
  AttributionControl,
} from "maplibre-gl";
import LocationSearch, { type LocationTarget } from "./LocationSearch";
import type {
  Map as MapLibreMap,
  ImageSource,
} from "maplibre-gl";
import type { ForecastData, ForecastFrame } from "@/lib/types";
import "maplibre-gl/dist/maplibre-gl.css";

/* =========================================================
   CONFIG
========================================================= */

const MAPTILER_KEY =
  process.env.NEXT_PUBLIC_MAPTILER_KEY;

/*
 * MapLibre worker configuration for Next.js/Turbopack.
 * The worker and shared module must exist at:
 * public/maplibre/maplibre-gl-worker.mjs
 * public/maplibre/maplibre-gl-shared.mjs
 */
setWorkerUrl(
  "/maplibre/maplibre-gl-worker.mjs"
);

const DOMAIN = {
  latMin: 28.05,
  latMax: 29.95,
  lonMin: 76.05,
  lonMax: 78.95,
};

const TRANSPARENT_PIXEL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

/* =========================================================
   TYPES
========================================================= */

/* =========================================================
   RAINFALL COLORS
========================================================= */

function rainfallColor(
  value: number
): [number, number, number, number] {
  if (!Number.isFinite(value) || value <= 0.01) {
    return [0, 0, 0, 0];
  }

  if (value < 0.1) {
    return [70, 190, 255, 75];
  }

  if (value < 1.0) {
    return [20, 130, 255, 115];
  }

  if (value < 2.5) {
    return [0, 90, 230, 150];
  }

  if (value < 5.0) {
    return [90, 40, 220, 175];
  }

  if (value < 10.0) {
    return [180, 40, 220, 195];
  }

  if (value < 25.0) {
    return [255, 120, 20, 215];
  }

  return [255, 35, 35, 225];
}

/* =========================================================
   MODEL GRID ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ IMAGEDATA
========================================================= */

function createRainfallImageData(
  grid: number[][]
): ImageData {
  const height = grid.length;
  const width = grid.reduce(
    (max, row) => Math.max(max, row?.length ?? 0),
    0,
  );

  if (!width || !height) {
    return new ImageData(1, 1);
  }

  const imageData = new ImageData(
    width,
    height
  );

  const pixels = imageData.data;

  for (
    let imageY = 0;
    imageY < height;
    imageY++
  ) {
    /*
     * Model row 0 = south
     * Image row 0 = north
     *
     * Therefore vertically flip.
     */
    const modelRow =
      height - 1 - imageY;

    for (
      let x = 0;
      x < width;
      x++
    ) {
      const value =
        grid[modelRow]?.[x] ?? 0;

      const [
        r,
        g,
        b,
        a,
      ] = rainfallColor(value);

      const pixelIndex =
        (imageY * width + x) * 4;

      pixels[pixelIndex] = r;
      pixels[pixelIndex + 1] = g;
      pixels[pixelIndex + 2] = b;
      pixels[pixelIndex + 3] = a;
    }
  }

  return imageData;
}

/* =========================================================
   COMPONENT
========================================================= */

export default function RainfallMap({
  forecast,
  frameIndex,
  loading = false,
  onFrameChange,
}: {
  forecast: ForecastData | null;
  frameIndex: number;
  loading?: boolean;
  onFrameChange: (index: number) => void;
  onLocationSelect?: (target: LocationTarget) => void;
}) {
  const mapContainer = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const mapLoadedRef = useRef(false);
  const sourceReadyRef = useRef(false);

  /* =======================================================
     UPDATE RAINFALL IMAGE
  ======================================================= */

  function buildRainfallFeatures(
    frame: ForecastFrame,
    gridBounds: ForecastData["grid"],
  ) {
    const height = frame.grid.length;
    const width = frame.grid[0]?.length ?? 0;

    if (!height || !width) {
      return [];
    }

    const latStep =
      (gridBounds.lat_max - gridBounds.lat_min) / height;

    const lonStep =
      (gridBounds.lon_max - gridBounds.lon_min) / width;

    const maxValue = Math.max(
      Number(frame.max) || 0,
      0.001,
    );

    const features = [];

    for (let row = 0; row < height; row++) {
      for (let col = 0; col < width; col++) {
        const rawValue =
          Number(frame.grid[row]?.[col] ?? 0);

        const value =
          Number.isFinite(rawValue) && rawValue > 0
            ? rawValue
            : 0;

        const lon =
          gridBounds.lon_min +
          (col + 0.5) * lonStep;

        const lat =
          gridBounds.lat_min +
          (row + 0.5) * latStep;

        const normalized =
          Math.min(1, value / maxValue);

        const weight =
          Math.pow(normalized, 0.55);

        features.push({
          type: "Feature",
          properties: {
            value,
            weight,
          },
          geometry: {
            type: "Point",
            coordinates: [
              lon,
              lat,
            ],
          },
        });
      }
    }

    return features;
  }
  function updateRainfallLayer(
    map: MapLibreMap,
    frame: ForecastFrame,
    gridBounds: ForecastData["grid"],
  ) {
    const source = map.getSource(
      "rainfall-forecast"
    ) as any;

    if (!source) {
      console.warn("Rainfall grid source not ready.");
      return;
    }

    const features = buildRainfallFeatures(
      frame,
      gridBounds,
    );

    source.setData({
      type: "FeatureCollection",
      features,
    });

    console.log(
      "RAINFALL GRID PATCHES UPDATED:",
      frame.horizon_minutes,
      "min",
      frame.min,
      "max",
      frame.max,
    );
  }
  /* =======================================================
     MAP INITIALIZATION
  ======================================================= */

  useEffect(() => {
    if (!mapContainer.current) {
      return;
    }

    if (!MAPTILER_KEY) {
      console.error(
        "NEXT_PUBLIC_MAPTILER_KEY is missing."
      );

      

      return;
    }

    console.log(
      "MAPTILER KEY FOUND"
    );

    const map =
      new Map({
        container:
          mapContainer.current,

        style:
          `https://api.maptiler.com/maps/hybrid/style.json?key=${MAPTILER_KEY}`,

        center: [
          77.7064,
          28.9845,
        ],

        zoom: 9.2,

        attributionControl: { compact: true },

        maxCanvasSize: [
          2048,
          2048,
        ],

        renderWorldCopies: false,
      });

    mapRef.current =
      map;

    /* =====================================================
       CONTROLS
    ===================================================== */

    map.addControl(
  new NavigationControl({
    showZoom: true,
    showCompass: true,
  }),
  "top-right"
);

map.addControl(
  new ScaleControl({
    maxWidth: 120,
    unit: "metric",
  }),
  "bottom-right"
);

    /* =====================================================
       DEBUG EVENTS
    ===================================================== */

    map.on(
      "style.load",
      () => {
        console.log(
          "MAP STYLE LOADED"
        );
      }
    );

    map.on(
      "error",
      (event) => {
        console.error("MAPLIBRE ERROR OBJECT:", event);
        console.error("MAPLIBRE ERROR:", event?.error);
        console.error("MAPLIBRE ERROR MESSAGE:", event?.error?.message);
      }
    );

    /* =====================================================
       MAP LOAD
    ===================================================== */

    map.on(
      "load",
      () => {
        console.log(
          "BASE MAP LOADED SUCCESSFULLY"
        );

        mapLoadedRef.current =
          true;

        /* =================================================
           CREATE RAINFALL GRID SOURCE
        ================================================= */

        try {
          map.addSource(
            "rainfall-forecast",
            {
              type: "geojson",
              data: {
                type: "FeatureCollection",
                features: [],
              },
            }
          );

          sourceReadyRef.current = true;

          console.log(
            "RAINFALL GRID SOURCE CREATED"
          );
        } catch (error) {
          console.error(
            "FAILED TO CREATE RAINFALL GRID SOURCE:",
            error
          );

          return;
        }

        /* =================================================
           CREATE RAINFALL GRID PATCH LAYER
        ================================================= */

        try {
          map.addLayer({
            id: "rainfall-forecast-layer",
            type: "heatmap",
            source: "rainfall-forecast",

            maxzoom: 16,

            paint: {
              "heatmap-weight": [
                "interpolate",
                ["linear"],
                ["get", "weight"],
                0, 0,
                0.15, 0.15,
                0.35, 0.35,
                0.60, 0.60,
                0.80, 0.82,
                1, 1,
              ],

              "heatmap-intensity": [
                "interpolate",
                ["linear"],
                ["zoom"],
                7, 0.8,
                11, 1.1,
                15, 1.35,
              ],

              "heatmap-radius": [
                "interpolate",
                ["linear"],
                ["zoom"],
                7, 22,
                10, 34,
                13, 48,
                15, 62,
                16, 70,
              ],

              "heatmap-opacity": [
                "interpolate",
                ["linear"],
                ["zoom"],
                7, 0.55,
                12, 0.68,
                16, 0.75,
              ],

              "heatmap-color": [
                "interpolate",
                ["linear"],
                ["heatmap-density"],
                0, "rgba(0,0,0,0)",
                0.08, "rgba(0,140,255,0.18)",
                0.20, "rgba(0,210,255,0.40)",
                0.38, "rgba(0,255,170,0.52)",
                0.56, "rgba(190,255,0,0.62)",
                0.72, "rgba(255,220,0,0.70)",
                0.84, "rgba(255,125,0,0.78)",
                0.94, "rgba(255,45,45,0.84)",
                1, "rgba(255,0,150,0.88)",
              ],
            },
          } as any);

          console.log(
            "RAINFALL GRID PATCH LAYER CREATED"
          );
        } catch (error) {
          console.error(
            "FAILED TO CREATE RAINFALL GRID LAYER:",
            error
          );

          return;
        }

        /* =================================================
           RENDER FIRST FRAME
        ================================================= */

        const data = forecast;

        if (
          data &&
          data.frames &&
          data.frames.length > 0
        ) {
          updateRainfallLayer(
            map,
            data.frames[0],
            data.grid,
          );
        }
        /* =================================================
           INITIAL RESIZE
        ================================================= */

        requestAnimationFrame(
          () => {
            map.resize();
          }
        );

        window.setTimeout(
          () => {
            map.resize();
          },
          100
        );

        window.setTimeout(
          () => {
            map.resize();
          },
          300
        );

        window.setTimeout(
          () => {
            map.resize();
          },
          700
        );
      }
    );

    /* =====================================================
       RESIZE OBSERVER
    ===================================================== */

    const resizeObserver =
      new ResizeObserver(
        (entries) => {
          const entry =
            entries[0];

          if (!entry) {
            return;
          }

          const width =
            entry.contentRect.width;

          const height =
            entry.contentRect.height;

          /*
           * The dashboard briefly reports
           * an 83px width during React/Next.js
           * layout reflow.
           *
           * Do NOT resize MapLibre to that
           * temporary width.
           */
          if (width < 300) {
            console.warn(
              "Ignoring transient map width:",
              width
            );

            return;
          }

          console.log(
            "MAP CONTAINER SIZE:",
            width,
            "x",
            height
          );

          map.resize();
        }
      );

    resizeObserver.observe(
      mapContainer.current
    );

    /* =====================================================
       CLEANUP
    ===================================================== */

    return () => {
      resizeObserver.disconnect();

      mapLoadedRef.current =
        false;

      sourceReadyRef.current =
        false;

      map.remove();

      mapRef.current =
        null;
    };
  }, []);

  /* =======================================================
     UPDATE FORECAST FRAME
  ======================================================= */

  useEffect(() => {
    const map =
      mapRef.current;

    if (!map) {
      return;
    }

    if (!forecast) {
      return;
    }

    if (!mapLoadedRef.current) {
      return;
    }

    if (!sourceReadyRef.current) {
      return;
    }

    const frame =
      forecast.frames[
        frameIndex
      ];

    if (!frame) {
      return;
    }

    updateRainfallLayer(
      map,
      frame,
      forecast.grid,
    );
  }, [
    forecast,
    frameIndex,
  ]);

  /* =======================================================
     CURRENT FRAME
  ======================================================= */

  const currentFrame =
    forecast?.frames?.[
      frameIndex
    ];

  const horizon =
    currentFrame?.horizon_minutes ??
    30;

  const mean =
    currentFrame?.mean ??
    0;

  /* =======================================================
     RENDER
  ======================================================= */

  return (
    <div
      className="
        relative
        h-full
        w-full
        min-h-0
        min-w-0
        overflow-hidden
        rounded-2xl
        border
        border-white/10
        bg-slate-950
      "
      style={{
        width: "100%",
        height: "100%",
        minWidth: 0,
        minHeight: 0,
      }}
    >
      {/* =================================================
          MAP
      ================================================= */}

      <div
        ref={mapContainer}
        className="
          absolute
          inset-x-0
          top-0
          bottom-[132px]
          block
          min-h-0
          min-w-0
        "
        style={{
          width: "100%",
          height: "100%",
          minWidth: 0,
          minHeight: 0,
        }}
      />
	<LocationSearch
  domain={
    forecast
      ? {
          latMin: forecast.grid.lat_min,
          latMax: forecast.grid.lat_max,
          lonMin: forecast.grid.lon_min,
          lonMax: forecast.grid.lon_max,
        }
      : undefined
  }
  resolutionKm={10}
  onLocationSelect={(target) => {
    const map = mapRef.current;

    if (!map) {
      return;
    }

    map.flyTo({
      center: [
        target.longitude,
        target.latitude,
      ],
      zoom: 15.5,
      duration: 1400,
      essential: true,
    });
  }}
  onOutsideDomain={() => {
    // Coverage notice is handled inside LocationSearch.
  }}
/>

      {/* =================================================
          FORECAST STATUS
      ================================================= */}

      <div
        className="
          absolute
          left-3
          top-3
          z-10
          rounded-xl
          border
          border-white/10
          bg-black/75
          px-3
          py-2.5
          shadow-xl
          backdrop-blur-md
        "
      >
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-cyan-400" />

          <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-300">
            Rainfall forecast
          </span>
        </div>

        <div className="mt-1 text-lg font-semibold text-white">
          +{horizon} min
        </div>

        <div className="mt-0.5 text-[10px] text-slate-400">
          Short-term rainfall intensity
        </div>

        <div className="mt-1.5 text-[9px] text-amber-300/80">
          Forecast coverage active
        </div>
      </div>

      {/* =================================================
          FORECAST DATA STATUS
      ================================================= */}

      <div
        className="
          absolute
          right-3
          top-3
          z-10
          rounded-lg
          border
          border-white/10
          bg-black/70
          px-2.5
          py-1.5
          backdrop-blur-md
        "
      >
        <div className="flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />

          <span className="text-[9px] font-medium uppercase tracking-wider text-slate-300">
            LIVE
          </span>
        </div>
      </div>

      {/* =================================================
          FORECAST TIMELINE
      ================================================= */}

      <div
        className="
          absolute
          bottom-0
          left-0
          right-0
          z-10
          rounded-none
          border-x-0
          border-b-0
          border-t
          border-white/10
          bg-black/85
          px-3
          py-2.5
          shadow-2xl
          backdrop-blur-md
        "
      >
        <div className="mb-2 flex items-center justify-between">
          <div>
            <div className="text-[11px] font-medium text-slate-200">
              Forecast timeline
            </div>

            <div className="text-[9px] text-slate-500">
              Rainfall forecast
            </div>
          </div>

          <div className="text-[10px] font-semibold text-cyan-300">
            {mean.toFixed(
              3
            )}{" "}
            mm/hr avg
          </div>
        </div>

        <input
          type="range"
          min={0}
          max={Math.max(
            0,
            (forecast?.frames?.length ??
              1) -
              1
          )}
          step={1}
          value={frameIndex}
          onChange={(event) =>
            onFrameChange(Number(event.target.value))
          }
          className="w-full accent-cyan-400"
        />

        <div className="mt-1.5 flex justify-between">
          {forecast?.frames?.map(
            (
              frame,
              index
            ) => (
              <button
                key={
                  frame.horizon_minutes
                }
                onClick={() =>
                  onFrameChange(index)
                }
                className={`
                  text-[8px]
                  transition
                  ${
                    index ===
                    frameIndex
                      ? "font-semibold text-cyan-300"
                      : "text-slate-500 hover:text-slate-300"
                  }
                `}
              >
                +
                {
                  frame.horizon_minutes
                }
                m
              </button>
            )
          )}
        </div>
      </div>

      {/* =================================================
          RAINFALL LEGEND
      ================================================= */}

      <div
        className="
          absolute
          bottom-[88px]
          right-3
          z-10
          rounded-lg
          border
          border-white/10
          bg-black/70
          px-2.5
          py-2
          backdrop-blur-md
        "
      >
        <div className="mb-1.5 text-[8px] font-semibold uppercase tracking-wider text-slate-400">
          Rainfall
        </div>

        <div className="flex items-center gap-1">
          <span className="text-[8px] text-slate-500">
            0
          </span>

          <div
            className="h-2 w-24 rounded-sm"
            style={{
              background:
                "linear-gradient(to right, rgba(70,190,255,.45), rgba(20,130,255,.75), rgba(0,90,230,.9), rgba(90,40,220,.95), rgba(180,40,220,1), rgba(255,120,20,1), rgba(255,35,35,1))",
            }}
          />

          <span className="text-[8px] text-slate-500">
            25+
          </span>

          <span className="text-[8px] text-slate-500">
            mm/hr
          </span>
        </div>
      </div>

      {/* =================================================
          LOADING
      ================================================= */}

      {loading && (
        <div
          className="
            absolute
            inset-0
            z-30
            flex
            items-center
            justify-center
            bg-slate-950/75
            backdrop-blur-sm
          "
        >
          <div
            className="
              rounded-xl
              border
              border-white/10
              bg-black/75
              px-5
              py-4
              text-center
            "
          >
            <div
              className="
                mx-auto
                mb-2
                h-5
                w-5
                animate-spin
                rounded-full
                border-2
                border-slate-600
                border-t-cyan-400
              "
            />

            <div className="text-xs text-slate-300">
              Loading forecastÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦
            </div>
          </div>
        </div>
      )}
    </div>
  );
}







