import os
from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
import logging
from dotenv import load_dotenv

load_dotenv()
# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# --- 1. Set up Logging and Environment Variables ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
TWILIO_AUTH_TOKEN = os.environ.get('AUTH_TOKEN')
TWILIO_ACCOUNT_SID = os.environ.get('ACCOUNT_SID')

if not all([GOOGLE_API_KEY, TWILIO_AUTH_TOKEN, TWILIO_ACCOUNT_SID]):
    logging.error("Missing required environment variables. Please set GOOGLE_API_KEY, AUTH_TOKEN, and ACCOUNT_SID.")
    exit()

# --- 2. Set up the LLM and RAG Knowledge Base with FAISS ---
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.2, api_key=GOOGLE_API_KEY)

dental_clinic_info = """
Clinic Name: Smile Center Dental Clinic
Location: 123 Main Street, Anytown, USA
Phone: (555) 123-4567
Hours: Monday-Friday, 9:00 AM - 5:00 PM. We are closed on weekends.
Services: Routine cleaning, fillings, root canals, teeth whitening, veneers.
Insurance: We accept all major dental insurance plans including Cigna, Delta Dental, and MetLife.
"""
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(dental_clinic_info)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)

docsearch = FAISS.from_texts(texts, embeddings)
retriever = docsearch.as_retriever()
rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# --- 3. Define Custom Tools for CrewAI ---
# Helper function to create a CrewAI-compatible tool from a LangChain StructuredTool
def create_crewai_tool_from_langchain_tool(langchain_tool):
    class CustomTool(BaseTool):
        name: str = langchain_tool.name
        description: str = langchain_tool.description

        def _run(self, *args, **kwargs):
            return langchain_tool.run(*args, **kwargs)
    return CustomTool()

@tool("Information Retriever")
def information_retriever(query: str) -> str:
    """
    Retrieves information from the dental clinic's knowledge base.
    Useful for answering questions about the clinic's address, hours, services, and insurance.
    Input should be a clear question about the clinic.
    """
    try:
        result = rag_chain.invoke({'query': query})['result']
        return result
    except Exception as e:
        logging.error(f"Error during information retrieval: {e}")
        return "I am sorry, I couldn't find that information at the moment."

# Tool to check calendar availability (placeholder)
class CalendarCheckTool(BaseTool):
    name: str = "Check Calendar Availability"
    description: str = "Checks the dentist's calendar for available appointment slots. Input is a requested date (e.g., 'October 15')."

    def _run(self, date: str) -> str:
        if "monday" in date.lower():
            return "Available slots on Monday: 10:00 AM, 2:00 PM, 3:30 PM."
        return f"No appointments available on {date}. Please try a different day."

# Tool to book an appointment (placeholder)
class AppointmentBookingTool(BaseTool):
    name: str = "Book Appointment"
    description:str = "Books a dental appointment. Input is the date, time, and patient name."

    def _run(self, data: str) -> str:
        return f"Appointment for {data} has been successfully booked. A confirmation SMS will be sent shortly."

# Instantiate the tools
calendar_tool = CalendarCheckTool()
booking_tool = AppointmentBookingTool()
# Create a CrewAI-compatible version of the Information Retriever tool
info_retriever_tool = create_crewai_tool_from_langchain_tool(information_retriever)

# --- 4. Define the CrewAI Agents and Tasks ---
# The Appointment Agent
booking_agent = Agent(
    role='Expert Dental Appointment Booker',
    goal='Accurately book appointments for patients based on their requests.',
    backstory='You are a friendly and efficient bot specialized in managing dental clinic schedules.',
    tools=[calendar_tool, booking_tool],
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# The Support Agent
support_agent = Agent(
    role='Dental Clinic Support Assistant',
    goal='Provide accurate and helpful information about the dental clinic.',
    backstory='You are a knowledgeable and polite assistant who provides information about the clinic.',
    tools=[info_retriever_tool],
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# The Booking Task
booking_task = Task(
    description='Analyze the user request to book an appointment and use the available tools to complete the task.',
    expected_output='A clear, friendly confirmation message of the booked appointment.',
    agent=booking_agent
)

# The Support Task
support_task = Task(
    description='Answer the user\'s question about the dental clinic using the provided information retriever tool.',
    expected_output='A concise and helpful answer to the user\'s question.',
    agent=support_agent
)

# --- 5. Set up the Flask App and Webhook ---
app = Flask(__name__)

# Initialize the Twilio request validator
validator = RequestValidator(TWILIO_AUTH_TOKEN)

@app.route("/whatsapp", methods=['POST'])
def whatsapp_webhook():
    # Retrieve the signature from the headers
    signature = request.headers.get('X-Twilio-Signature')
    
    # Use request.form which is a MultiDict object, ideal for the validator
    # if not validator.validate(request.url, request.form, signature):
    #     logging.warning("Invalid Twilio signature. Request rejected.")
    #     return "Invalid signature", 403

    incoming_msg = request.form.get('Body', '').lower()
    from_number = request.form.get('From', '')

    resp = MessagingResponse()
    msg = resp.message()

    try:
        # Intent Classification using the LLM
        intent_prompt = f"""
        Classify the user's intent based on the following message. 
        Respond with only the classification word.
        Classifications:
        - `book_appointment`: The user wants to schedule an appointment.
        - `support_question`: The user is asking for general information about the clinic (e.g., hours, location, services).
        - `unknown`: The intent is unclear or irrelevant.

        User message: '{incoming_msg}'
        """
        intent = llm.invoke(intent_prompt).content.strip().lower()

        logging.info(f"User message from {from_number}: {incoming_msg}")
        logging.info(f"Classified intent: {intent}")

        bot_reply = "I'm sorry, I didn't understand that. Please ask about booking an appointment or for general clinic information."

        if 'book_appointment' in intent:
            crew = Crew(
                agents=[booking_agent],
                tasks=[booking_task],
                verbose=2,
                process=Process.sequential
            )
            result = crew.kickoff(inputs={'user_request': incoming_msg})
            bot_reply = result

        elif 'support_question' in intent:
            crew = Crew(
                agents=[support_agent],
                tasks=[support_task],
                verbose=2,
                process=Process.sequential
            )
            result = crew.kickoff(inputs={'user_request': incoming_msg})
            bot_reply = result

    except Exception as e:
        logging.error(f"An error occurred while processing the request: {e}", exc_info=True)
        bot_reply = "I apologize, I am having a technical issue. Please try again in a few moments."

    msg.body(bot_reply)
    return str(resp)

if __name__ == "__main__":
    logging.info("Starting Flask application...")
    app.run(debug=True, port=5000)