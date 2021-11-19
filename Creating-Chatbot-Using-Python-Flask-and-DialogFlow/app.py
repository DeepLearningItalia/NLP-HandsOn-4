import os

import dialogflow
from flask import Flask, request, jsonify, render_template
from google.protobuf.json_format import MessageToDict

from transformers import pipeline
import tensorflow as tf

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")


app = Flask(__name__)


def detect_intent_with_parameters(project_id, session_id, query_params, language_code, user_input):
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(project_id, session_id)
    text = user_input
    text_input = dialogflow.types.TextInput(text=text, language_code=language_code)
    query_input = dialogflow.types.QueryInput(text=text_input)
    response = session_client.detect_intent(session=session, query_input=query_input, query_params=query_params)
    return response


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/chat', methods=["Post"])
def chat():
    print("Entering Chat Mode")
    input_text = request.form['message']

    GOOGLE_AUTHENTICATION_FILE_NAME = "credentials/key.json"
    current_directory = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(current_directory, GOOGLE_AUTHENTICATION_FILE_NAME)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

    GOOGLE_PROJECT_ID = "small-talk-jcnv"#"pizza-ordering-dsik"
    session_id = "1234567891"
    context_short_name = "does_not_matter"

    context_name = "projects/" + GOOGLE_PROJECT_ID + "/agent/sessions/" + session_id + "/contexts/" + context_short_name.lower()

    parameters = dialogflow.types.struct_pb2.Struct()

    context_1 = dialogflow.types.context_pb2.Context(
        name=context_name,
        lifespan_count=2,
        parameters=parameters
    )
    query_params_1 = {"contexts": [context_1]}

    language_code = 'en'

    response = detect_intent_with_parameters(
        project_id=GOOGLE_PROJECT_ID,
        session_id=session_id,
        query_params=query_params_1,
        language_code=language_code,
        user_input=input_text
    )
    print("*******RESPOSE")
    print(response)
    result = MessageToDict(response)
    print("\n\n******RESULT")
    print(result)
    if len(result['queryResult']['fulfillmentMessages']) == 2:
        response = {"message": result['queryResult']['fulfillmentText'],
                    "payload": result['queryResult']['fulfillmentMessages'][1]['payload']}
    else:
        response = {"message": result['queryResult']['fulfillmentText']+" - Input Sentiment: "+ sentiment_analysis(input_text)[0]["label"] , "payload": None}
    # response = {"message": result['queryResult']['fulfillmentText'], "payload": None}
    return jsonify(response)


if __name__ == '__main__':
    app.run()
