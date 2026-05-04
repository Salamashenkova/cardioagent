from gigachat import GigaChat
from gigachat.models import Chat

# ТВОЙ СЫРОЙ токен
gc = GigaChat(
    credentials="MTUyNzYzZDctZWZjOS00NjdlLTkxN2UtODhhMWU3ZDVkYzgzOmRiMGZmMTllLWVjNjYtNGNkOS05ZmQ4LTU1ZTE2ZDVhODAyZQ==",
    scope="GIGACHAT_API_PERS",
    verify_ssl_certs=False
)

try:
    resp = gc.chat(Chat(messages=[{"role": "user", "content": "Привет"}]))
    print("✅ GIGACHAT РАБОТАЕТ:", resp.choices[0].message.content)
except Exception as e:
    print("❌ ОШИБКА:", e)
