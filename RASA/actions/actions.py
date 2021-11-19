# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
from pprint import pprint

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

from transformers import pipeline
import tensorflow as tf

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")

class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Hello World!")

        return []

class ActionBookHotel(Action):

    def name(self) -> Text:
        return "action_book_hotel"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        hotels = [
            {
                "name": "Hotel 1",
                "review": "It is very bad"
            },
            {
                "name": "Hotel 2",
                "review": "It is very good"
            }
        ]

        dispatcher.utter_message(text="Custom Action!")
        
        print("** DOMAIN **")
        pprint(domain)

        print("\n\n** TRACKER ** ")
        pprint(tracker.events)

        print("\n\n** Latest Message **")
        pprint(tracker.latest_message)

        result = None
        for h in hotels:
            sent = sentiment_analysis(h["review"])[0]["label"]
            if sent == "POSITIVE":
                result = h["name"]
                break
        
        dispatcher.utter_message(text="Hotel: "+str(result))

        return [SlotSet("hotel", result if result is not None else [])]
