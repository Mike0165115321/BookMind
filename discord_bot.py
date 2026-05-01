import discord
import httpx
import asyncio

from collections import defaultdict

TOKEN = 'MTQ5OTI5OTcyODExNjk0NTA0Ng.GHUshM.OCM5WopBeGZeLTc-9lA3adwfGSbxstPmuDL7N0'

# Session Memory: {user_id: [messages]}
memory = defaultdict(list)
MAX_MEMORY = 6  # เก็บ 3 คู่ (ถาม-ตอบ)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

def determine_mode(query: str) -> str:
    """Smart Router: Decide mode based on keywords."""
    keywords = ['วิเคราะห์', 'เจาะลึก', 'เปรียบเทียบ', 'สรุป', 'ขั้นตอน', 'ทำอย่างไร', 'ทำไม', 'เชื่อมโยง', 'ลึกๆ']
    if any(k in query for k in keywords):
        return 'agentic'
    return 'classic'

@client.event
async def on_ready():
     print(f'✅ Bot พร้อมใช้งาน: {client.user}')

@client.event
async def on_message(message):
     if message.author.bot:
          return

     user_id = str(message.author.id)
     user_query = message.content
     
     # 1. Determine Mode (Smart Router)
     mode = determine_mode(user_query)
     
     # 2. Build History Context
     history = memory[user_id]
     context_query = user_query
     if history:
          history_str = "\n".join(history[-MAX_MEMORY:])
          context_query = f"บริบทการสนทนาก่อนหน้า:\n{history_str}\n\nคำถามปัจจุบัน: {user_query}"

     async with message.channel.typing():
          try:
               async with httpx.AsyncClient(timeout=120) as http:
                    response = await http.post(
                         'http://localhost:8080/api/ask',
                         json={
                              'query': context_query,
                              'use_hyde': True,
                              'mode': mode
                         }
                    )

               # Parse SSE response
               full_answer = ''
               for line in response.text.split('\n'):
                    if line.startswith('data:'):
                         import json
                         try:
                              data = json.loads(line[5:].strip())
                              if 'text' in data:
                                   full_answer += data['text']
                         except:
                              pass

               reply = full_answer or 'ไม่สามารถตอบได้ในขณะนี้'

               # 3. Send Response in Chunks (Discord limit 2000 chars)
               chunk_size = 1900
               for i in range(0, len(reply), chunk_size):
                    chunk = reply[i:i + chunk_size]
                    if i == 0:
                         await message.reply(chunk)
                    else:
                         await message.channel.send(chunk)

               # 3. Update Memory
               memory[user_id].append(f"User: {user_query}")
               memory[user_id].append(f"AI: {reply[:2000]}") # จำไว้สูงสุด 2000 ตัวอักษรต่อข้อความ
               if len(memory[user_id]) > MAX_MEMORY:
                    memory[user_id] = memory[user_id][-MAX_MEMORY:]

          except Exception as e:
               print(f'Error: {e}')
               await message.reply('❌ ระบบขัดข้อง กรุณาลองใหม่')

client.run(TOKEN)   