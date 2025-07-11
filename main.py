from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch


load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.8)

positive_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("user", "give a thank you message to this positive feedback: {input}")
    ]
)

negative_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("user", "give a thank you message to this negative feedback: {input}")
    ]
)

neutral_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("user", "give a thank you message to this neutral feedback: {input}")
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("user", "escalate this feedback to the manager: {input}")
    ]
)


# Classification prompt for user feedback
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),
    ]
)

# First classify the sentiment, then branch based on the classification
branches = classification_template | llm | StrOutputParser() | RunnableBranch(
    (
        lambda x: "positive" in str(x).lower(), positive_prompt | llm | StrOutputParser(),
    ),
    (
        lambda x: "negative" in str(x).lower(),
        negative_prompt | llm | StrOutputParser(),
    ),
    (
        lambda x: "neutral" in str(x).lower(),
        neutral_prompt | llm | StrOutputParser(),
    ),
    escalate_feedback_template | llm | StrOutputParser(),
)


# review = "The product is great! I love the features and the design is beautiful."
review = "The product is not good. I hate the features and the design is ugly."

chain = classification_template | llm | StrOutputParser() | branches

print(chain.invoke({"feedback":review}))
