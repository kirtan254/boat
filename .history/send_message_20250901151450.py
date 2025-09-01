from twilio.rest import Client

# Your Account SID and Auth Token from twilio.com/console
# Be sure to store these securely and NOT hardcode them in production.
account_sid = 'ACb644facd5bb5b056efad68f1b863eeba'  # Replace with your Account SID
auth_token = 'your82366a3d61872327d2845e5b6ff9f80d_auth_token_here'              # Replace with your Auth Token
client = Client(account_sid, auth_token)

# Your WhatsApp sandbox number (from the Twilio console)
from_whatsapp_number = 'whatsapp:+14155238886'

# Your personal WhatsApp number (in E.164 format)
to_whatsapp_number = 'whatsapp:+919173321399' # Replace with your number

message = client.messages.create(
    body='Hello from your new bot! This is a test message.',
    from_=from_whatsapp_number,
    to=to_whatsapp_number
)

print(message.sid)