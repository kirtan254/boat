from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

@app.route("/whatsapp", methods=['POST'])
def whatsapp_webhook():
    # Get the message body from the incoming request
    incoming_msg = request.values.get('Body', '').lower()

    # Create a TwiML response
    resp = MessagingResponse()
    msg = resp.message()

    # Simple "echo" logic
    if 'hello' in incoming_msg:
        msg.body('Hello there! How can I help you today?')
    else:
        msg.body('I received your message! I am a basic echo bot for now. Stay tuned for my upgrades!')

    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)