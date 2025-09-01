from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

@app.route("/whatsapp", methods=['POST'])
def whatsapp_webhook():
    # Print the full payload of both form and values
    print("Full form payload:", request.form)
    print("Full values payload:", request.values)

    # Get the message body from the incoming request
    incoming_msg = request.form.get('Body', '').lower()

    # This is how you see the user's message
    print(f"Received message from user: {incoming_msg}")
    print(f"I received an empty message: {incoming_msg == ''}")

    # Create a TwiML response
    resp = MessagingResponse()
    msg = resp.message()

    # Your reply logic
    if 'hello' in incoming_msg:
        msg.body('Hello there! How can I help you today?')
    else:
        msg.body('I received your message! I am a basic echo bot for now. Stay tuned for my upgrades!')
    
    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)