
import os
from videosdk.agents import Agent, AgentSession, Pipeline, JobContext, RoomOptions, WorkerJob
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.elevenlabs import ElevenLabsTTS

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

# Pre-downloading the Turn Detector model
pre_download_model()

class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="""You are a knowledgeable and approachable AI voice agent for utilities. Your primary role is to assist users with their utility services, including electricity, water, and gas inquiries. Your key responsibilities include providing information on billing, usage, and service-related questions, helping users understand their utility statements, and guiding them through troubleshooting any service issues. You should maintain a clear, courteous, and helpful tone, ensuring users feel supported and informed. Be proactive in suggesting energy-saving tips or offering solutions to common problems. While you are an expert in utility management and services, do not handle personal or payment information directly; instead, direct users to the secure customer portal for such transactions.""")
    async def on_enter(self): await self.session.say("Hello! How can I help you today regarding ai voice agent for utilities?")
    async def on_exit(self): await self.session.say("Goodbye!")

async def start_session(context: JobContext):
    # Create agent
    agent = MyVoiceAgent()

    # Create pipeline
    pipeline = Pipeline(
        stt=DeepgramSTT(model="nova-2", language="en"),
        llm=OpenAILLM(model="gpt-4o"),
        tts=ElevenLabsTTS(model="eleven_flash_v2_5"),
        vad=SileroVAD(threshold=0.35),
        turn_detector=TurnDetector(threshold=0.8)
    )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(
     #  room_id="<room_id>",  # Set to join a pre-created room; omit to auto-create
        name="VideoSDK Cascaded Agent for ai voice agent for utilities",
        playground=True
    )

    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start()
