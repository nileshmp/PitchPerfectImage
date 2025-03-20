def generate_prompt(image_path):
    prompt = f"""
You are an image quality evaluator. Your task is to score the provided image on a scale of 0 to 10.

*   **10:** Perfect still image. No video player artifacts.
*   **0:**  Clearly a screenshot of a video with prominent video player controls.

Penalize the image *heavily* for any of the following:

*   **Progress bars/Timelines:** Any horizontal bar indicating video progress.
*   **Play/Pause Buttons:**  Triangular "play" symbols, or "pause" symbols (two vertical bars).
*   **Volume Controls:** Speaker icons or volume sliders.
*   **Timestamps:**  Text displaying the video's current time or duration (e.g., "0:35 / 2:15").
*   **Full-Screen Buttons:**  Icons indicating full-screen mode.
*   **Other UI Elements:** Any other visual elements that are clearly part of a video player interface.

Provide a score and a brief explanation justifying your score.

Image: (Image will be inserted here programmatically)

Score:
Explanation:
"""
    return prompt

# --- Example Usage (Conceptual, requires integration with an LLM API) ---
import openai # or other LLM library

def score_image(image_path, api_key):
    prompt = generate_prompt(image_path)
    # Load and encode the image (Base64 encoding is common for APIs)
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the API request (This is a simplified example, adapt to your LLM API)
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview", # Or your chosen model
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                            "detail": "low" # Adjust detail level if needed
                        },
                    },
                ],
            }
        ],
        max_tokens=300, # Adjust as needed
    )

    # Extract the score and explanation from the API response
    llm_response = response.choices[0].message.content
    print(llm_response) # For debugging
    # You'll need to parse the `llm_response` string to extract the score and explanation.
    # This might involve using regular expressions or string manipulation.
    try:
      score = int(llm_response.split("Score:")[1].split("\n")[0].strip())
    except:
      score = None # Handle parsing errors gracefully
    try:
        explanation = llm_response.split("Explanation:")[1].strip()
    except:
        explanation = ""

    return score, explanation

import base64
# --- Example ---
if __name__ == '__main__':
    # Replace with your actual API key and image path
    api_key = "YOUR_OPENAI_API_KEY"  
    image_path = "path/to/your/image.jpg"

    score, explanation = score_image(image_path, api_key)

    if score is not None:
        print(f"Image Score: {score}")
        print(f"Explanation: {explanation}")
    else:
        print("Error: Could not get a score from the LLM.")