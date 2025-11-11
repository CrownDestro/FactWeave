// src/components/ResultCard.tsx
"use client";
import React from "react";
import Meter from "./Meter";

interface ResultCardProps {
  label: string;
  fakeProbability: number;
  realProbability: number;
}

export default function ResultCard({ label, fakeProbability, realProbability }: ResultCardProps) {
  const percentage = label === "Fake" ? fakeProbability : realProbability;

  return (
    <div className="mt-8 flex flex-col items-center gap-4 bg-gray-800/50 rounded-2xl p-6 shadow-lg w-full border border-gray-700">
      <Meter percentage={percentage} label={label} />
      <div className="mt-4 text-center text-sm text-gray-400 space-y-1">
        <p>🧠 Fake Probability: {fakeProbability.toFixed(2)}%</p>
        <p>✅ Real Probability: {realProbability.toFixed(2)}%</p>
      </div>
    </div>
  );
}