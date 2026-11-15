# Oasis
**Track:** AI + Health
**Team:** Elijah Burke & Jacob Pool

## The problem
Today, many people, especially in the younger generations, struggle with numerous mental, emotional, and often spiritual issues. If not properly handled, these issues could hamper the futures set before them, potentially leading to unhealthy outbursts of emotion or even suicidal thoughts. This project aims to bring easily accessible emotional health guidance and report changes in a user’s emotional well-being over time. 
## What we built
We built a model-agnostic LLM harness in Python that records by conversation the user's mental state throughout a given day, showing a graph when enough data has been collected and encouraging, offering sympathy & empathy where appropriate and promoting self-reflection. One would simply type their thoughts to this software at least once per day using natural language.
## How it works
This program works by exposing specific tools to an AI model with instructions on its purpose within the program. The data follows a specific JSON format as created by tools.py, meaning datasets or specially trained models are not required for proper function.
This project was tested against Qwen3.5-4B with thinking disabled and a context length of 8192 hosted from LM Studio. Qwen3.5-4B is a small yet surprisingly capable LLM that can run on many consumer hardware configurations.
## Results
The success of this project cannot yet be measured quantitatively. We believe this project reached the proof-of-concept stage, with decent results in the tested configuration.
The instruction-following capabilities of models under 9B parameters are very inconsistent, making local models usable but not always reliable for this program. Using a larger model--specifically cloud-hosted frontier models--would likely achieve great results.

## What we'd do next
With more time, we'd further fine-tune and optimize the system prompts and tool instructions to ensure consistent results and efficient usage. We would also address the rendering and interface for a more polished user experience, as well as supporting more inference provider endpoints.
## Sources
This project was created with assistance from Anthropic's Claude Sonnet 5.5, OpenAI's GPT-5.6 Sol, and OpenAI's GPT-6 Luna.
This project was tested against Qwen3.5-4B with thinking disabled and a context length of 8192 hosted from LM Studio.

## Disclaimer
As a proof-of-concept project, this project can produce unexpected or undesired responses. Use with discretion.
