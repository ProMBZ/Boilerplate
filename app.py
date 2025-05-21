# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from agents import Agent, Runner, Tool, input_guardrail, output_guardrail, GuardrailFunctionOutput, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered, RunContextWrapper
from pydantic import BaseModel
from tavily import TavilyClient
import requests
import google.generativeai as genai

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Gemini setup
genai.configure(api_key=GEMINI_API_KEY)

def fallback_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini error: {e}"

def fallback_serper(query):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY}
    data = {"q": query}
    try:
        response = requests.post(url, json=data, headers=headers)
        return response.json().get("organic", [])[0].get("snippet", "No result found")
    except Exception as e:
        return f"Serper error: {e}"

# Tavily tool
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

def tavily_search(query):
    try:
        result = tavily_client.search(query)
        return result.get("results", [{}])[0].get("content", "No result found")
    except:
        return fallback_serper(query) or fallback_gemini(query)

search_tool = Tool(name="tavily_search", description="Search the web for recent info.", func=tavily_search)

# Weather tool
def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    try:
        res = requests.get(url).json()
        return f"{res['name']} - {res['weather'][0]['description']}, {res['main']['temp']}°C"
    except:
        return "Weather data not available"

weather_tool = Tool(name="get_weather", description="Get weather info for a city.", func=get_weather)

# News tool
def get_news(topic):
    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={NEWS_API_KEY}"
    try:
        res = requests.get(url).json()
        return res['articles'][0]['description']
    except:
        return "No news found"

news_tool = Tool(name="get_news", description="Get recent news about a topic.", func=get_news)

# Guardrails
class GuardrailOutput(BaseModel):
    blocked: bool
    reason: str

guardrail_agent = Agent(
    name="Guardrail",
    instructions="Tell if the input is inappropriate, unsafe or unrelated.",
    output_type=GuardrailOutput,
)

@input_guardrail
async def content_guard(ctx: RunContextWrapper, agent: Agent, input):
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(output_info=result.final_output, tripwire_triggered=result.final_output.blocked)

class OutputCheck(BaseModel):
    is_math: bool
    reasoning: str

output_guard_agent = Agent(
    name="OutputGuard",
    instructions="Check if the output contains math answers.",
    output_type=OutputCheck,
)

@output_guardrail
async def prevent_math(ctx: RunContextWrapper, agent: Agent, output):
    result = await Runner.run(output_guard_agent, output, context=ctx.context)
    return GuardrailFunctionOutput(output_info=result.final_output, tripwire_triggered=result.final_output.is_math)

# Main agent
main_agent = Agent(
    name="Main Assistant",
    instructions="You are a helpful AI assistant.",
    tools=[search_tool, weather_tool, news_tool],
    input_guardrails=[content_guard],
    output_guardrails=[prevent_math],
)

# Streamlit UI
st.set_page_config(page_title="AI Agent Boilerplate", layout="centered")
st.title("🤖 AI Agent Boilerplate")

query = st.text_input("Ask something:")
if st.button("Run") and query:
    with st.spinner("Thinking..."):
        try:
            result = Runner.run_sync(main_agent, query)
            st.success(result.final_output)
        except InputGuardrailTripwireTriggered:
            st.error("❌ Input blocked by guardrail: inappropriate or unsafe content.")
        except OutputGuardrailTripwireTriggered:
            st.warning("⚠️ Output was blocked due to containing math answers.")
        except Exception as e:
            fallback = fallback_gemini(query)
            st.info(f"🤖 Fallback (Gemini): {fallback}")
