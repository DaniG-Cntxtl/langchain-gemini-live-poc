import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI

#from langgraph.pregel import create_agent_executor # This import needs to be adjusted based on your actual agent setup.
print("List of models that support generateContent:\n")
genai.configure(api_key="AIzaSyC9vifSNF_RB6hOaWL59ikUk1-6r2AT0uQ")
for m in genai.list_models():
    # 'm.supported_actions' replaces the deprecated 'm.supported_methods'
    # if "generateContent" in m.supported_actions:
    print(m.name)

# Initialize the Gemini model
# Note: The 'native-audio' model (gemini-2.5-flash-native-audio-preview-09-2025) only supports the 'bidiGenerateContent' (real-time streaming) API,
# which is not yet supported by the standard LangChain GoogleGenerativeAI class (which uses 'generateContent').
# We are using the standard preview model instead.
llm = GoogleGenerativeAI(model="gemini-2.5-flash-preview-09-2025",
                             api_key="AIzaSyC9vifSNF_RB6hOaWL59ikUk1-6r2AT0uQ")

# --- Direct Invocation Test ---
# This checks that your API key is correctly set up and the model can be called.
try:
    messages = [HumanMessage(content="what is the weather in India")]
    response = llm.invoke(messages)
    print("Direct Gemini invocation successful:", response)
except Exception as e:
    print(f"Direct Gemini invocation failed: {e}")