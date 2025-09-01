from twilio.rest import Client
import os
from dotenv import load_dotenv

load_dotenv()

account_sid = os.getenv("ACCOUNT_SID")
auth_token = os.getenv("AUTH_TOKEN")
client = Client(account_sid, auth_token)

from_whatsapp_number = 'whatsapp:+14155238886'
to_whatsapp_number = 'whatsapp:+919173321399'

message = client.messages.create(
    body='Hello from your new bot! This is a test message.',
    from_=from_whatsapp_number,
    to=to_whatsapp_number
)

print(message.sid)
