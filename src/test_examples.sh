#!/bin/bash

echo "🧠 Running Fake News Prediction Verification Tests"
echo "====================================================="
echo ""

# Function to run test and format output
run_test() {
    echo "📰 Text: $1"
    echo "---------------------------------------------"
    python3 src/predict_news.py "$1" | grep -E "Fake|Real|🎯|✅|🧩"
    echo ""
}

# PolitiFact
run_test "Barack Obama bans the Pledge of Allegiance in all U.S. schools."
run_test "The Affordable Care Act was signed into law by President Obama in 2010."

# GossipCop
run_test "Jennifer Aniston and Brad Pitt are getting married again in secret ceremony."
run_test "Taylor Swift released her album 1989 (Taylor’s Version) in 2023."

# CoAID
run_test "Drinking hot water every 15 minutes kills COVID-19 instantly."
run_test "The World Health Organization declared COVID-19 a pandemic in March 2020."

# PolitiFact (Extra)
run_test "Joe Biden sold U.S. secrets to China according to leaked documents."
run_test "Kamala Harris became the first female Vice President of the United States in 2021."

# GossipCop (Extra)
run_test "Kanye West is running for president again in 2025 with Elon Musk."

# CoAID (Extra)
run_test "COVID-19 vaccines cause infertility in young women."

echo "====================================================="
echo "✅ Verification tests completed!"
