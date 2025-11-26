import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

# -------------------------
# CONFIG
# -------------------------

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "models/gemini-flash-latest"

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=API_KEY)

app = Flask(__name__)
CORS(app)


# -------------------------
# ROUTE: AI ASSISTANT
# -------------------------

@app.route("/intelli-ai", methods=["POST"])
def intelli_ai():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        history = data.get("history", "").strip()

        if not user_message:
            return jsonify({"reply": "No message received from app."})

        print(f"[AI] Incoming from app: {user_message}")
        print(f"[AI] History length: {len(history)} characters")

        prompt = f"""
        You are Intelli, the AI assistant inside a messaging app called IntelliChat.
        You must answer based on the conversation history and the latest user message.

        Here is the conversation so far (from oldest to newest):
        {history}

        Now the user says: {user_message}

        Respond as Intelli in a friendly, helpful, and concise way. Do not repeat the full history.
        """

        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)

        reply_text = getattr(response, "text", None)
        if not reply_text:
            if hasattr(response, "candidates") and response.candidates:
                try:
                    reply_text = response.candidates[0].content.parts[0].text
                except Exception:
                    reply_text = "AI had no response."
            else:
                reply_text = "AI had no response."

        print(f"[AI] Reply: {reply_text[:80]}...")
        return jsonify({"reply": reply_text})

    except Exception as e:
        print(f"[AI] ERROR: {e}")
        return jsonify({"reply": f"AI error: {str(e)}"}), 500


# -------------------------
# HEALTH CHECK ROUTE
# -------------------------

@app.route("/", methods=["GET"])
def health():
    return "Intelli AI Server is running.", 200


# -------------------------
# ENTRY POINT
# -------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Running locally on port {port}")
    app.run(host="0.0.0.0", port=port)
