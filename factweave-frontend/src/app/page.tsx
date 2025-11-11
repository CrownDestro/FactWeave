// app/page.tsx
"use client";

import React, { useState } from "react";
import ResultCard from "../components/ResultCard";

interface ApiResponse {
  text: string;
  fake_probability: number;
  real_probability: number;
  predicted_label: string;
}

const API_URL = "http://127.0.0.1:5000/predict";

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handlePredict = async () => {
    setError("");
    if (!text.trim()) {
      setError("Please enter a news statement.");
      return;
    }

    try {
      setLoading(true);
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!res.ok) throw new Error("Server error");
      const data: ApiResponse = await res.json();
      setResult(data);
    } catch (err) {
      setError("❌ Could not reach the prediction server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6 bg-gray-900">
      <main className="flex flex-col items-center w-full max-w-2xl">
        <h1 className="text-4xl font-bold text-indigo-400 mb-8 text-center">
          📰 Fake News Detector
        </h1>

        <textarea
          className="w-full h-32 resize-none mb-4 bg-gray-800 border border-gray-700 rounded-lg p-4 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          placeholder="Enter news text here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <button
          onClick={handlePredict}
          disabled={loading}
          className={`w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 ${
            loading ? "opacity-60 cursor-not-allowed" : ""
          }`}
        >
          {loading ? "Analyzing..." : "Detect Fake News"}
        </button>

        {error && <p className="mt-4 text-red-400 text-sm text-center">{error}</p>}
        {result && (
          <ResultCard
            label={result.predicted_label}
            fakeProbability={result.fake_probability}
            realProbability={result.real_probability}
          />
        )}
      </main>
    </div>
  );
}