import cv2 as cv
import streamlit as st
from streamlit_webrtc import webrtc_streamer,WebRtcMode,VideoProcessorBase
from turn import get_ice_servers
# import av
# import copy
# import mediapipe as mp

st.set_page_config(page_title="Basketball")

st.header("Basketball video edit")

import os
from twilio.rest import Client

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
client = Client(account_sid, auth_token)

token = client.tokens.create()

webrtc_ctx = webrtc_streamer(
    key="basketball",
    # mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": token.ice_servers},
)
st.session_state["started"] = webrtc_ctx.state.playing
