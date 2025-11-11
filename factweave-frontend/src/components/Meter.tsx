// src/components/Meter.tsx
"use client";
import React from "react";

interface MeterProps {
  percentage: number;
  label: string;
}

export default function Meter({ percentage, label }: MeterProps) {
  const color =
    label === "Fake" ? "text-red-400 border-red-500" : "text-green-400 border-green-500";

  return (
    <div className="flex flex-col items-center">
      <div
        className={`relative flex items-center justify-center w-40 h-40 rounded-full border-8 ${color} transition-all duration-700`}
        style={{
          borderColor: label === "Fake" ? "#ef4444" : "#22c55e",
          background: "rgba(255,255,255,0.05)",
        }}
      >
        <span className="text-2xl font-bold">{percentage.toFixed(1)}%</span>
      </div>
      <p className={`mt-3 text-xl font-semibold ${color}`}>{label}</p>
    </div>
  );
}