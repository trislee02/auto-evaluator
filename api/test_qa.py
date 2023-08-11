import os
import logging
from dotenv import load_dotenv
from lib.QAGenerator import QAGenerator
from json import JSONDecodeError

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()
API_BASE = os.environ.get("API_BASE")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
CHAT_ENDPOINT = os.environ.get("CHAT_ENDPOINT")
GPT_ENGINE = os.environ.get("GPT_ENGINE")

qa_generator = QAGenerator(API_BASE, AZURE_OPENAI_KEY, GPT_ENGINE)

text = """Now lets see how we can change a voice (language) of QTrobot. Following are some standard supported languages:

*  **en-US** (American English)
*  **fr-FR** (French)
*  **de-DE** (German)

You may have different languages installed on your QTrobot. This tutorial will use English and French language.

Open the `tutorial_qt_speech_node.py` file and add the following code:

```python
#!/usr/bin/env python
import sys
import rospy
from std_msgs.msg import String
from qt_robot_interface.srv import *

if __name__ == '__main__':
    rospy.init_node('my_tutorial_node')
    rospy.loginfo("my_tutorial_node started!")

    # define a ros service
    speechConfig = rospy.ServiceProxy('/qt_robot/speech/config', speech_config)

    # define a ros service
    speechSay = rospy.ServiceProxy('/qt_robot/speech/say', speech_say)
    
    # block/wait for ros service
    rospy.wait_for_service('/qt_robot/speech/say') 
    
    # block/wait for ros service
    rospy.wait_for_service('/qt_robot/speech/config') 
   
    try:
        status = speechConfig("en-US",0,0)
        if status:
            speechSay("Hello, I am speaking English")
            status = False
        rospy.sleep(1)
        status = speechConfig("fr-FR",0,0)
        if status:
            speechSay("Bonjour, Je parle français")
        

    except KeyboardInterrupt:
        pass

    rospy.loginfo("finsihed!")

```

## Explanation

ROS Services are defined by srv files, which contains a request message and a response message. First we import all from `qt_robot_interface.srv`. This will import all srv files that are under `qt_robot_interface.srv`. We need to use `speech_config`. """

# Catch any QA generation errors and re-try until QA pair is generated
awaiting_answer = True
retry_count = 0
max_retry = 3
while awaiting_answer:
    try:
        retry_count += 1
        qa_pair = qa_generator.run(text, logger)
        logger.info(qa_pair)
        break
    except JSONDecodeError:
        logger.error("Error on question")
        if retry_count >= max_retry:
            break