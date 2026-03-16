#!/usr/bin/env python3
"""
NBA Betting Model - Web Dashboard
Run: python server.py
Then open http://localhost:5000
"""

import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from queue import Empty, Queue

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

# Optional: Load .env for OPENAI_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")

# ── Paths (all relative to project root) ─────────────────────────────────────
DETAILED_JSON_PATH = "results/todays_bets_detailed.json"
BACKTEST_PATH = "results/backtest_log.csv"
PIPELINE_SCRIPT = "main.py"

# ────────────────────────────────────────────────────────────────────────────
# Pages
# ────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ────────────────────────────────────────────────────────────────────────────
# API – Predictions
# ────────────────────────────────────────────────────────────────────────────

@app.route("/api/predictions")
def get_predictions():
    if not os.path.exists(DETAILED_JSON_PATH):
        return jsonify({"error": "No predictions found. Run the pipeline first.", "predictions": [], "date": None})
    try:
        with open(DETAILED_JSON_PATH) as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e), "predictions": [], "date": None}), 500


# ────────────────────────────────────────────────────────────────────────────
# API – Backtest
# ────────────────────────────────────────────────────────────────────────────

@app.route("/api/backtest")
def get_backtest():
    if not os.path.exists(BACKTEST_PATH):
        return jsonify({"error": "Backtest log not found.", "data": []})
    try:
        df = pd.read_csv(BACKTEST_PATH)
        df = df.dropna(subset=["Date", "Bankroll"])
        # Sample to ~300 points so the chart is snappy
        if len(df) > 300:
            step = max(1, len(df) // 300)
            df = df.iloc[::step]
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

        wins = int((df["Result"] == "WIN").sum())
        losses = int((df["Result"] == "LOSS").sum())
        total = wins + losses
        roi = float(((df["Bankroll"].iloc[-1] - 1000) / 1000) * 100) if len(df) else 0

        return jsonify({
            "labels": df["Date"].tolist(),
            "bankroll": df["Bankroll"].round(2).tolist(),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total * 100, 1) if total else 0,
            "roi": round(roi, 1),
            "total_bets": total,
        })
    except Exception as e:
        return jsonify({"error": str(e), "data": []}), 500


# ────────────────────────────────────────────────────────────────────────────
# API – Run Pipeline (Server-Sent Events streaming)
# ────────────────────────────────────────────────────────────────────────────

_pipeline_lock = threading.Lock()
_pipeline_running = False


@app.route("/api/run-pipeline")
def run_pipeline():
    global _pipeline_running

    mode = request.args.get("mode", "predict")  # predict | full

    if _pipeline_lock.locked():
        def already_running():
            yield "data: ⚠️  Pipeline is already running. Please wait.\n\n"
            yield "data: __DONE__\n\n"
        return Response(stream_with_context(already_running()), mimetype="text/event-stream")

    def generate():
        global _pipeline_running
        with _pipeline_lock:
            _pipeline_running = True
            try:
                if mode == "full":
                    cmd = [sys.executable, PIPELINE_SCRIPT]
                else:
                    # Fast mode: skip data collection & training, just regenerate predictions
                    cmd = [sys.executable, "src/models/predict_today.py"]

                yield f"data: 🚀 Starting {'full pipeline' if mode == 'full' else 'prediction'}...\n\n"

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in iter(proc.stdout.readline, ""):
                    clean = line.rstrip()
                    if clean:
                        yield f"data: {clean}\n\n"
                proc.wait()
                if proc.returncode == 0:
                    yield "data: ✅ Done!\n\n"
                else:
                    yield f"data: ❌ Process exited with code {proc.returncode}\n\n"
            except Exception as e:
                yield f"data: ❌ Error: {e}\n\n"
            finally:
                _pipeline_running = False
                yield "data: __DONE__\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ────────────────────────────────────────────────────────────────────────────
# API – AI Analysis
# ────────────────────────────────────────────────────────────────────────────

def _build_analysis_prompt(game: dict) -> str:
    home = game.get("Home", "")
    away = game.get("Away", "")
    pick = game.get("Model_Pick", "")
    win_pct = game.get("Win_Prob", "")
    edge = game.get("Edge", "")
    action = game.get("Action", "")
    pot_profit = game.get("Pot_Profit", "")

    h_elo = game.get("Home_ELO", 0)
    a_elo = game.get("Away_ELO", 0)
    h_elo_raw = game.get("Home_ELO_Raw", 0)
    a_elo_raw = game.get("Away_ELO_Raw", 0)
    h_rs = game.get("Home_Roster_Strength", 0)
    a_rs = game.get("Away_Roster_Strength", 0)
    h_rs_max = game.get("Home_Roster_Max", 0)
    a_rs_max = game.get("Away_Roster_Max", 0)
    h_inj = game.get("Home_Injuries", [])
    a_inj = game.get("Away_Injuries", [])
    h_b2b = game.get("Home_IS_B2B", 0)
    a_b2b = game.get("Away_IS_B2B", 0)
    h_3in4 = game.get("Home_IS_3IN4", 0)
    a_3in4 = game.get("Away_IS_3IN4", 0)
    h_off = game.get("Home_OFF_RTG", 0)
    a_off = game.get("Away_OFF_RTG", 0)
    h_def = game.get("Home_DEF_RTG", 0)
    a_def = game.get("Away_DEF_RTG", 0)
    h_efg = game.get("Home_EFG_PCT", 0)
    a_efg = game.get("Away_EFG_PCT", 0)
    h_ml = game.get("Home_ML")
    a_ml = game.get("Away_ML")

    def fmt_ml(ml):
        if ml is None:
            return "N/A"
        return f"+{ml}" if ml > 0 else str(ml)

    h_health = round(h_rs / h_rs_max * 100, 1) if h_rs_max else 100
    a_health = round(a_rs / a_rs_max * 100, 1) if a_rs_max else 100

    inj_home_str = ", ".join(h_inj) if h_inj else "None"
    inj_away_str = ", ".join(a_inj) if a_inj else "None"

    schedule_home = []
    if h_b2b:
        schedule_home.append("back-to-back")
    if h_3in4:
        schedule_home.append("3-in-4 schedule")
    schedule_away = []
    if a_b2b:
        schedule_away.append("back-to-back")
    if a_3in4:
        schedule_away.append("3-in-4 schedule")

    return f"""You are an expert NBA betting analyst. Provide a concise, insightful analysis (3-5 short paragraphs) explaining why the model favors {pick} in this matchup. Be specific, data-driven and engaging. Do NOT just repeat the numbers — interpret what they mean.

MATCHUP: {home} (home) vs {away} (away)
MODEL PICK: {pick} | Win Probability: {win_pct} | Edge vs Vegas: {edge}
BETTING ACTION: {action} | Potential Profit: {pot_profit}

ELO RATINGS (injury-adjusted):
  {home}: {h_elo} (raw: {h_elo_raw})
  {away}: {a_elo} (raw: {a_elo_raw})

ROSTER HEALTH (RAPTOR-based active roster vs full strength):
  {home}: {h_rs:.1f} / {h_rs_max:.1f} ({h_health}% health)
  {away}: {a_rs:.1f} / {a_rs_max:.1f} ({a_health}% health)

INJURIES:
  {home} out: {inj_home_str}
  {away} out: {inj_away_str}

SCHEDULE FATIGUE:
  {home}: {', '.join(schedule_home) if schedule_home else 'Fresh'}
  {away}: {', '.join(schedule_away) if schedule_away else 'Fresh'}

RECENT FORM (EWMA-10):
  Offensive Rating — {home}: {h_off}, {away}: {a_off}
  Defensive Rating — {home}: {h_def}, {away}: {a_def}
  eFG% — {home}: {h_efg:.1%}, {away}: {a_efg:.1%}

VEGAS MONEYLINES:
  {home}: {fmt_ml(h_ml)} | {away}: {fmt_ml(a_ml)}

Focus your analysis on: the key competitive advantage, any injury/health impact, schedule factors, and whether the edge is legitimate or narrow. End with one sentence summarising the confidence level."""


@app.route("/api/analyze", methods=["POST"])
def analyze_game():
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        return jsonify({
            "analysis": "⚠️ OpenAI API key not configured. Add OPENAI_API_KEY to a .env file in the project root to enable AI analysis.",
            "source": "fallback",
        })

    try:
        from openai import OpenAI
    except ImportError:
        return jsonify({
            "analysis": "⚠️ openai package not installed. Run: pip install openai",
            "source": "fallback",
        })

    game = request.get_json()
    if not game:
        return jsonify({"error": "No game data provided"}), 400

    try:
        client = OpenAI(api_key=openai_key)
        prompt = _build_analysis_prompt(game)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a sharp NBA betting analyst. Be concise, data-driven, and insightful. Use plain text — no markdown headers or bullet points."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.7,
        )
        analysis = response.choices[0].message.content.strip()
        return jsonify({"analysis": analysis, "source": "openai"})
    except Exception as e:
        return jsonify({"analysis": f"⚠️ AI analysis failed: {str(e)}", "source": "error"}), 500


# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NBA Betting Model — Web Dashboard")
    print("  Open: http://localhost:8080")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=8080, threaded=True)
