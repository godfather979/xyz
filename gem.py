from google import genai

client = genai.Client(api_key="AIzaSyDHqAMZtrzCThMGO8mmD-JGEPPvMOlpfqE")

#gemini-2.0-flash
#gemini-2.5-flash
#gemini-2.5-pro

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Give just python simple code for water jug problem using BFS algorithm. No comments. Give sample code run also",
)

print(response.text)